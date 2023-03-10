import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytest import fixture


@fixture(scope="session")
def tetrahedra():
    from tetranerf import cpp

    data = Path(__file__).absolute().parent / "assets" / "bottle.ply"
    mesh = trimesh.load(str(data))
    cells = cpp.triangulate(torch.from_numpy(mesh.vertices).float())
    return {"vertices": mesh.vertices.astype(np.float32), "cells": cells.numpy()}


def generate_rays(width=800, height=800):
    def normalize(x):
        return x / torch.linalg.norm(x, dim=-1, keepdim=True)

    m_eye = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32)
    m_lookat = torch.tensor((0, 0, 0), dtype=torch.float32)
    m_up = torch.tensor((0, 0, 1), dtype=torch.float32)
    m_fovY = 45.0
    m_aspectRatio = width / height

    W = m_lookat - m_eye
    wlen = torch.linalg.norm(W, dim=-1)
    U = normalize(torch.linalg.cross(W, m_up))
    V = normalize(torch.linalg.cross(U, W))

    vlen = wlen * math.tan(0.5 * m_fovY * math.pi / 180.0)
    V *= vlen
    ulen = vlen * m_aspectRatio
    U *= ulen

    d = torch.stack(
        tuple(
            reversed(
                torch.meshgrid(
                    torch.linspace(0, 1, width),
                    torch.linspace(0, 1, height),
                    indexing="ij",
                )
            )
        ),
        -1,
    )
    d = 2.0 * d - 1.0
    d = d.view(-1, 2)
    ray_directions = normalize(d[:, :1] * U[None] + d[:, 1:] * V[None] + W[None])
    ray_origins = torch.repeat_interleave(m_eye[None], len(ray_directions), 0)
    return ray_origins, ray_directions


def test_traversal_ray_tracing(tmp_path, tetrahedra):
    from tetranerf import cpp

    ray_origins, ray_directions = generate_rays(400, 400)

    device = torch.device("cuda:0")
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)
    cuda_vertices = torch.from_numpy(tetrahedra["vertices"]).float().to(device)
    cuda_cells = torch.from_numpy(tetrahedra["cells"]).int().to(device)

    tracer = cpp.TetrahedraTracer(device)
    tracer.load_tetrahedra(cuda_vertices, cuda_cells)
    ellapsed = 0
    out = None
    for _ in range(20):
        if out is not None:
            del out
            torch.cuda.empty_cache()
            time.sleep(0.50)
        start = time.time()
        out = tracer.trace_rays(
            ray_origins,
            ray_directions,
            200,
        )
        end = time.time()
        ellapsed += end - start
    print(f"trace_rays time: {ellapsed/20}")
    ray_samples = (
        torch.linspace(0.90, 1.1, 300, dtype=torch.float32, device=device)
        .expand((out["num_visited_cells"].shape[0], 300))
        .contiguous()
    )
    start = time.time()
    inter_out = None
    for _ in range(20):
        if inter_out is not None:
            del inter_out
        inter_out = tracer.find_visited_cells(
            out["num_visited_cells"],
            out["visited_cells"],
            out["barycentric_coordinates"],
            out["hit_distances"],
            ray_samples,
        )
    end = time.time()
    print(f"find_visited_cells time: {(end - start)/20}")
    torch.set_printoptions(linewidth=200, sci_mode=False)
    print(inter_out["mask"].shape)
    print(inter_out["mask"].max())
    print(inter_out["mask"].int().sum(-1).argmax())
    print(inter_out["mask"][56607].int())
    mask = torch.arange(out["hit_distances"].shape[1])[None, :] < out["num_visited_cells"][:, None].cpu()
    space = (out["hit_distances"][:, 1:, 0] - out["hit_distances"][:, :-1, 1]).abs().cpu() * mask[:, 1:]
    ray_max, ray_min = (
        out["hit_distances"][:, :, 1].cpu().max(1)[0],
        (out["hit_distances"][:, :, 0].cpu() + (1 - mask.int()) * 10e4).min(1)[0],
    )
    print(out["hit_distances"][:, :, 0].cpu()[mask > 0].min())
    ray_len = (
        out["hit_distances"][:, :, 1].cpu().max(1)[0]
        - (out["hit_distances"][:, :, 0].cpu() + (1 - mask.int()) * 10e4).min(1)[0]
    )
    print("min: ", ray_min[ray_len > 0].min(), "max:", ray_max[ray_len > 0].max())
    valid_samples = (
        torch.logical_and(ray_min[:, None] < ray_samples.cpu(), ray_samples.cpu() < ray_max[:, None]).int().sum(-1)
    )
    actual_samples = (
        (
            torch.logical_and(
                ray_min[:, None] < ray_samples.cpu(),
                ray_samples.cpu() < ray_max[:, None],
            ).int()
            * inter_out["mask"].cpu().int()
        )
        .int()
        .sum(-1)
    )
    print("invalid ray portion", ((space.sum(-1) / (ray_len + 1e-4)))[ray_len > 0].mean())
    print(
        "invalid ray samples",
        (1 - (actual_samples / (valid_samples + 1e-4)))[ray_len > 0].mean(),
    )

    (nonempty_rays,) = np.where(inter_out["mask"].int().sum(-1).cpu().numpy() > 0)
    points = []
    pc = []
    random.seed(42)
    rays = random.sample(list(nonempty_rays), 100)
    nfaces = []
    # rays = [56607]
    nnpointsx = []
    print(rays)
    # rays = [97810]

    ray = 97810
    print(actual_samples[ray], "/", valid_samples[ray])
    print(inter_out["mask"][ray].int())
    print(
        ((out["hit_distances"][ray, 1:, 0] - out["hit_distances"][ray, :-1, 1]).abs().cpu() * mask[ray, 1:]).sum(-1),
        "/",
        ray_len[ray],
    )

    for r in rays:
        for c in range(out["num_visited_cells"][r]):
            endpoints = tetrahedra["vertices"][tetrahedra["cells"][out["visited_cells"][r, c].cpu().numpy(), :], :]
            coords = out["barycentric_coordinates"][r, c].cpu().numpy()
            # print(coords)
            v1, v2 = coords @ endpoints
            l = np.linspace(0, 1, 100)[:, None]
            points.append(v1 * l + v2 * (1 - l))
            pc.append(np.full((100, 4), 255, dtype=np.uint8))
            cell2 = tetrahedra["cells"][out["visited_cells"][r, c].cpu().numpy(), :]
            cell2 = np.concatenate((cell2, cell2), -1)
            nfaces.append(
                np.stack(
                    [
                        cell2[1:4],
                        cell2[2:5],
                        cell2[3:6],
                        cell2[4:7],
                    ]
                )
            )

        # Test interpolated points
        endpoints = tetrahedra["cells"][inter_out["cell_indices"][r][inter_out["mask"][r]].cpu().numpy()]
        nnpoints = tetrahedra["vertices"][endpoints]
        mults = inter_out["barycentric_coordinates"][r][inter_out["mask"][r]].cpu().numpy()
        nnpoints = (nnpoints * mults[..., None]).sum(-2)
        nnpointsx.append(nnpoints)
    print(nnpointsx[0].shape)
    trimesh.PointCloud(vertices=np.concatenate(nnpointsx, 0)).export(str(tmp_path / "intepolated_points.ply"))

    # points.append(ray_origins.cpu().numpy()[r:r+1] + np.linspace(0.8, 1, 200)[:, None] @ ray_directions.cpu().numpy()[r:r+1])
    # pc.append(np.array([[255, 0, 0, 255]] * 200, dtype=np.uint8))
    # points = np.concatenate(points, 0)
    # pc = np.concatenate(pc, 0)
    # trimesh.Trimesh(tetrahedra["vertices"], np.concatenate(nfaces, 0)).export("test_mesh.ply")
    # pcc = trimesh.PointCloud(points, pc)
    # pcc.export("test_points.ply")


def test_tetrahedra_interpolate_values(tetrahedra):
    from tetranerf import cpp
    from tetranerf.utils.extension import interpolate_values

    ray_origins, ray_directions = generate_rays(400, 400)

    device = torch.device("cuda:0")
    num_rays = 4096
    ray_origins = ray_origins.to(device)[80200 : num_rays + 80200]
    ray_directions = ray_directions.to(device)[80200 : num_rays + 80200]
    cuda_vertices = torch.from_numpy(tetrahedra["vertices"]).float().to(device)
    cuda_cells = torch.from_numpy(tetrahedra["cells"]).int().to(device)

    tracer = cpp.TetrahedraTracer(device)
    tracer.load_tetrahedra(cuda_vertices, cuda_cells)

    out = tracer.trace_rays(
        ray_origins,
        ray_directions,
        200,
    )
    num_ray_samples = 256
    ray_samples = (
        torch.linspace(0.90, 1.1, num_ray_samples, dtype=torch.float32, device=device)
        .expand((num_rays, num_ray_samples))
        .contiguous()
    )
    inter_out = tracer.find_visited_cells(
        out["num_visited_cells"],
        out["visited_cells"],
        out["barycentric_coordinates"],
        out["hit_distances"],
        ray_samples,
    )

    # Save some momory
    del out
    torch.cuda.empty_cache()
    time.sleep(0.1)

    # Interpolation forward
    num_vertices = len(tetrahedra["vertices"])
    field = torch.empty((num_vertices, 64), dtype=torch.float32, device=device).random_()
    ellapsed = 0
    for i in range(20):
        start = time.time()
        val = interpolate_values(
            inter_out["vertex_indices"],
            inter_out["barycentric_coordinates"],
            field,
        )
        end = time.time()
        ellapsed += end - start
        assert val.shape == (num_rays, num_ray_samples, 64)
        if i == 0:
            print("fingerprint: ", val.sum())
            # test
            gt = torch.einsum(
                "rbij,rbi->rj",
                field[inter_out["vertex_indices"].long()],
                inter_out["barycentric_coordinates"],
            )
            torch.testing.assert_allclose(val.sum(1), gt)
        del val
    print(f"forward time: {(ellapsed)/20}")

    # Interpolation backward
    start = time.time()
    ellapsed = 0
    field.requires_grad_(True)
    for i in range(20):
        val = interpolate_values(
            inter_out["vertex_indices"],
            inter_out["barycentric_coordinates"],
            field,
        )
        start = time.time()
        field.grad = None
        val.sum().backward()
        assert field.grad is not None
        grad = field.grad
        end = time.time()
        ellapsed += end - start
        assert grad.shape == (num_vertices, 64)
        if i == 0:
            pass
            # print("fingerprint: ", val.sum())
            # # test
            field2 = field.detach().clone()
            field2.requires_grad_(True)
            gt = torch.einsum(
                "rbij,rbi->rj",
                field2[inter_out["vertex_indices"].long()],
                inter_out["barycentric_coordinates"],
            )
            assert gt.requires_grad
            gt.sum().backward()
            assert field2.grad is not None
            torch.testing.assert_allclose(field2.grad, grad)
        del val
        del grad
    print(f"backward time: {(ellapsed)/20}")

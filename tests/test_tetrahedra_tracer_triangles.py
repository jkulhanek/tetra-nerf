import math
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
    tmp_path = Path(".")
    from tetranerf import cpp

    # ray_origins, ray_directions = generate_rays(400, 400)
    ray_origins, ray_directions = generate_rays(64, 64)

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
        out = tracer.trace_rays_triangles(
            ray_origins,
            ray_directions,
            256,
        )
        end = time.time()
        ellapsed += end - start
    print(f"trace_rays time: {ellapsed/20}")
    start = time.time()
    torch.set_printoptions(linewidth=200, sci_mode=False)
    mask = torch.arange(out["hit_distances"].shape[1])[None, :] < out["num_visited_triangles"][:, None].cpu()

    (nonempty_rays,) = np.where(out["num_visited_triangles"].int().cpu().numpy() > 0)
    points = []
    pc = []
    random.seed(42)
    rays = random.choices(list(nonempty_rays), k=100)
    # rays = [56607]
    nnpointsx = []
    # rays = [97810]
    # ray = 97810

    mask = torch.arange(out["hit_distances"].shape[1])[None, :] < out["num_visited_triangles"][:, None].cpu()
    for r in rays:
        for c in range(out["num_visited_triangles"][r]):
            endpoints = tetrahedra["vertices"][out["vertex_indices"][r, c].cpu().numpy(), :]
            coords = out["barycentric_coordinates"][r, c]
            coords = torch.cat((1-coords.sum(-1, keepdim=True), coords), -1).cpu().numpy()
            v = coords @ endpoints
            points.append(v)
            pc.append(np.full((1, 4), 255, dtype=np.uint8))

        # Test interpolated points
        endpoints = out["vertex_indices"][r][mask[r]].cpu().numpy()
        nnpoints = tetrahedra["vertices"][endpoints]
        gt_barycentric_coords = torch.cat((1-out["barycentric_coordinates"].sum(-1, keepdim=True), out["barycentric_coordinates"]), -1)
        mults = gt_barycentric_coords[r][mask[r]].cpu().numpy()
        nnpoints = (nnpoints * mults[..., None]).sum(-2)
        nnpointsx.append(nnpoints)

        dirs = nnpoints - ray_origins[r].cpu().numpy()
        dots = np.dot(dirs / np.linalg.norm(dirs, axis=-1, keepdims=True), ray_directions[r].cpu().numpy())
        if np.any(np.abs(dots - 1.0) > 0.05):
            raise RuntimeError("Points not projected onto the ray")
    print(nnpointsx[0].shape)
    trimesh.PointCloud(vertices=np.concatenate(nnpointsx, 0)).export(str(tmp_path / "intepolated_points.ply"))


def mix_float3(data, m, *args):
    k = data * m
    if len(args) > 0:
        k = k + mix_float3(*args)
    return k


def test_trace_rays_simple():
    from tetranerf import cpp
    device = torch.device("cuda:0")
    tetrahedra_points = torch.tensor([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [0.5, 0.5, 0.5]], dtype=torch.float32)

    tetrahedra_cells = torch.tensor([
        [0, 1, 2, 8],
        [2, 1, 3, 8],
        [0, 1, 4, 8],
        [4, 1, 5, 8],
        [0, 2, 4, 8],
        [4, 2, 6, 8],
        [4, 5, 6, 8],
        [5, 6, 7, 8],
        [2, 3, 6, 8],
        [3, 6, 7, 8],
        [1, 3, 5, 8],
        [3, 5, 7, 8]], dtype=torch.int32)

    origins = torch.tensor([[-0.05, 0.05, 0.05]], dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    d_tetrahedra_points = tetrahedra_points.to(device)
    d_tetrahedra_cells = tetrahedra_cells.to(device)
    d_origins = origins.to(device)
    d_directions = directions.to(device)

    tracer = cpp.TetrahedraTracer(device)
    tracer.load_tetrahedra(d_tetrahedra_points, d_tetrahedra_cells)
    out = tracer.trace_rays_triangles(d_origins, d_directions, 16)

    # TODO: check results


def test_find_tetrahedra():
    from tetranerf import cpp
    device = torch.device("cuda:0")
    tetrahedra_points = torch.tensor([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [0.5, 0.5, 0.5]], dtype=torch.float32)

    tetrahedra_cells = torch.tensor([
        [0, 1, 2, 8],
        [2, 1, 3, 8],
        [0, 1, 4, 8],
        [4, 1, 5, 8],
        [0, 2, 4, 8],
        [4, 2, 6, 8],
        [4, 5, 6, 8],
        [5, 6, 7, 8],
        [2, 3, 6, 8],
        [3, 6, 7, 8],
        [1, 3, 5, 8],
        [3, 5, 7, 8]], dtype=torch.int32)


    points = torch.stack([
        mix_float3(
            tetrahedra_points[0], 0.23,
            tetrahedra_points[1], 0.27,
            tetrahedra_points[2], 0.21,
            tetrahedra_points[8], 0.29),
    mix_float3(
        tetrahedra_points[2], 0.23,
        tetrahedra_points[4], 0.24,
        tetrahedra_points[6], 0.26,
        tetrahedra_points[8], 0.27),
    mix_float3(
        tetrahedra_points[3], 0.39,
        tetrahedra_points[5], 0.41,
        tetrahedra_points[7], 0.09,
        tetrahedra_points[8], 0.11)], 0)

    d_tetrahedra_points = tetrahedra_points.to(device)
    d_tetrahedra_cells = tetrahedra_cells.to(device)
    d_points = points.to(device)

    tracer = cpp.TetrahedraTracer(device)
    tracer.load_tetrahedra(d_tetrahedra_points, d_tetrahedra_cells)

    out = tracer.find_tetrahedra(d_points)

    gt_coords = torch.tensor([
            [0.23, 0.27, 0.21, 0.29],
            [0.23, 0.24, 0.26, 0.27],
            [0.39, 0.41, 0.09, 0.11]
        ], dtype=torch.float32)
    gt_indices = torch.tensor([
        [0, 1, 2, 8],
        [2, 4, 6, 8],
        [3, 5, 7, 8]], dtype=torch.int32)

    assert torch.all(out["tetrahedra"].cpu() == torch.tensor([0, 5, 11], dtype=torch.int32))
    bt_coords = out["barycentric_coordinates"].cpu()
    bt_coords = torch.cat((1-bt_coords.sum(-1, keepdim=True), bt_coords), -1)
    for i in range(len(points)):
        indices, _ind = torch.sort(out["vertex_indices"][i].cpu())

        coords = bt_coords[i][_ind].cpu()
        assert torch.all(indices == gt_indices[i])
        torch.testing.assert_allclose(
            coords,
            gt_coords[i])

def test_tetrahedra_interpolate_values(tetrahedra):
    from tetranerf import cpp
    from tetranerf.utils.extension import interpolate_values

    ray_origins, ray_directions = generate_rays(400, 400)
    # ray_origins, ray_directions = generate_rays(400, 400)

    device = torch.device("cuda:0")
    # num_rays = 4096
    num_rays = 256
    ray_origins = ray_origins[80200 : num_rays + 80200].to(device)
    ray_directions = ray_directions[80200 : num_rays + 80200].to(device)
    cuda_vertices = torch.from_numpy(tetrahedra["vertices"]).float().to(device)
    cuda_cells = torch.from_numpy(tetrahedra["cells"]).int().to(device)

    tracer = cpp.TetrahedraTracer(device)
    tracer.load_tetrahedra(cuda_vertices, cuda_cells)

    max_triangles = 256
    out = tracer.trace_rays_triangles(
        ray_origins,
        ray_directions,
        max_triangles,
    )

    # Save some momory
    torch.cuda.empty_cache()
    time.sleep(0.1)

    # Interpolation forward
    num_vertices = len(tetrahedra["vertices"])
    field = torch.empty((64, num_vertices), dtype=torch.float32, device=device).random_()
    ellapsed = 0
    vi = out["vertex_indices"]
    safe_vi = vi.long().clamp_max(field.size(-1))
    def get_field_safe(field):
        return torch.where(vi >= 0, field[:, safe_vi], torch.zeros_like(field[:, safe_vi]))
    for i in range(20):
        start = time.time()
        val = interpolate_values(
            vi,
            out["barycentric_coordinates"],
            field,
        )
        end = time.time()
        ellapsed += end - start
        assert val.shape == (num_rays, max_triangles, 64)
        if i == 0:
            print("fingerprint: ", val.sum())
            # test
            gt_barycentric_coords = torch.cat((1-out["barycentric_coordinates"].sum(-1, keepdim=True), out["barycentric_coordinates"]), -1)
            gt = torch.einsum(
                "jrbi,rbi->rbj",
                get_field_safe(field),
                gt_barycentric_coords,
            )
            torch.testing.assert_allclose(val, gt)
        del val
    print(f"forward time: {(ellapsed)/20}")

    # Interpolation backward
    start = time.time()
    ellapsed = 0
    field.requires_grad_(True)
    for i in range(20):
        val = interpolate_values(
            out["vertex_indices"],
            out["barycentric_coordinates"],
            field,
        )
        start = time.time()
        field.grad = None
        val.sum().backward()
        assert field.grad is not None
        grad = field.grad
        end = time.time()
        ellapsed += end - start
        assert grad.shape == (64, num_vertices)
        if i == 0:
            # print("fingerprint: ", val.sum())
            # # test
            field2 = field.detach().clone()
            field2.requires_grad_(True)
            gt_barycentric_coords = torch.cat((1-out["barycentric_coordinates"].sum(-1, keepdim=True), out["barycentric_coordinates"]), -1)
            gt = torch.einsum(
                "jrbi,rbi->rj",
                get_field_safe(field2),
                gt_barycentric_coords,
            )
            assert gt.requires_grad
            gt.sum().backward()
            assert field2.grad is not None
            torch.testing.assert_allclose(field2.grad, grad)
        del val
        del grad
    print(f"backward time: {(ellapsed)/20}")

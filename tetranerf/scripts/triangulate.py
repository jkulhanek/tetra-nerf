from pathlib import Path

import numpy as np
import torch
import trimesh
import tyro

from ..utils.extension import cpp
from .utils import CONSOLE, status


def entrypoint(
    pointcloud: Path,
    output: Path = Path("tetrahedra.th"),
    max_pointcloud_size: int = 1000000,
    random_points_ratio: float = 0.0,
    use_gaussian: bool = False,
):
    pc = trimesh.load(pointcloud)
    vertices = pc.vertices
    colors = pc.visual.vertex_colors
    if colors is not None:
        CONSOLE.print(f"Loaded {len(vertices)} points [green bold]with colors")
    else:
        CONSOLE.print(f"Loaded {len(vertices)} points [red bold]without colors")

    if len(vertices) > max_pointcloud_size:
        CONSOLE.print(f"Subsampling to {max_pointcloud_size} points")
        indices = np.random.permutation(len(vertices))[:max_pointcloud_size]
        vertices = vertices[indices]
        if colors is not None:
            colors = colors[indices]

    average_spacing = cpp.find_average_spacing(torch.from_numpy(vertices).float())
    CONSOLE.print(f"Average point spacing is {average_spacing:.4f}")

    if random_points_ratio > 0.0:
        std = average_spacing * 5
        num_samples = int(len(vertices) * random_points_ratio)
        CONSOLE.print(f"Sampling additional {num_samples} points with std={std:.4f}")
        base_indices = np.random.choice(len(vertices), num_samples, replace=True)
        new_vertices = vertices[base_indices]
        if use_gaussian:
            random_offsets = np.random.normal(0, std, size=(num_samples, 3))
        else:
            random_offsets = np.random.normal(0, 1, size=(num_samples, 3))
            random_offsets /= np.linalg.norm(random_offsets, axis=-1, keepdims=True)
            random_offsets *= np.abs(np.random.normal(average_spacing, average_spacing * 0.5, size=(num_samples, 1)))
        new_vertices = new_vertices + random_offsets  # np.random.normal(0, std, size=(num_samples, 3))
        vertices = np.concatenate((vertices, new_vertices), 0)
        if colors is not None:
            new_colors = colors[base_indices]
            new_colors[-1] = 0
            colors = np.concatenate((colors, new_colors), 0)

    # pointcloud_path = output.absolute().with_suffix(".ply")
    # if pointcloud_path != pointcloud.absolute():
    #     trimesh.PointCloud(
    #         vertices=vertices,
    #         colors=colors,
    #     ).export(pointcloud_path)
    #     CONSOLE.print(f"Sampled pointcloud saved to [bold]{pointcloud_path}")

    with status("[yellow] Running triangulation..."):
        cells = cpp.triangulate(torch.from_numpy(vertices).float())
        CONSOLE.print(f"[green bold]Triangulation finished with {len(cells)} tetrahedra")

    out = {
        "cells": cells,
        "vertices": torch.from_numpy(vertices).float(),
    }
    if colors is not None:
        out["colors"] = torch.from_numpy(colors)
    output.absolute().parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output.absolute()))
    CONSOLE.print(f"Triangulation saved to [bold]{output.absolute()}")


if __name__ == "__main__":
    tyro.cli(entrypoint)

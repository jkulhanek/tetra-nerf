import os
from pathlib import Path

import numpy as np
import torch
import trimesh


def test_triangulation(tmp_path):
    from tetranerf import cpp

    data = Path(__file__).absolute().parent / "assets" / "bottle.ply"
    mesh = trimesh.load(str(data))
    cells = cpp.triangulate(torch.from_numpy(mesh.vertices).float())
    assert len(cells.shape) == 2
    assert cells.shape[-1] == 4
    assert cells.max() == len(mesh.vertices) - 1
    assert len(cells) > 2500
    faces = torch.cat(
        (
            cells[:, 1:],
            torch.roll(cells, -1, -1)[:, 1:],
            torch.roll(cells, -2, -1)[:, 1:],
            torch.roll(cells, -3, -1)[:, 1:],
        )
    )
    trimesh.Trimesh(mesh.vertices, faces.numpy()).export(str(tmp_path / "tetrahedra.ply"))
    np.savez(str(tmp_path / "tetrahedra.npz"), cells=cells.numpy(), vertices=mesh.vertices)


def test_find_average_spacing():
    from tetranerf.utils.extension import cpp

    points = torch.linspace(0, 99, 100, dtype=torch.float32)[:, None] @ torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float32
    )
    spacing = cpp.find_average_spacing(points)
    assert 1 < spacing < 10

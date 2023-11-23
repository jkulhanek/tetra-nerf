from typing import Any, TypedDict
from pathlib import Path
from torch import Tensor

import torch

cpp = Any


class _FindVisitedCellsResult(TypedDict):
    cell_indices: Tensor
    vertex_indices: Tensor
    mask: Tensor
    barycentric_coordinates: Tensor


class _TraceRaysResult(TypedDict):
    num_visited_cells: Tensor
    visited_cells: Tensor
    barycentric_coordinates: Tensor
    hit_distances: Tensor
    vertex_indices: Tensor


class _TraceRaysTrianglesResult(TypedDict):
    num_visited_triangles: Tensor
    visited_triangles: Tensor
    barycentric_coordinates: Tensor
    hit_distances: Tensor
    vertex_indices: Tensor


class TetrahedraTracer:
    def __init__(self, device: torch.device, /) -> None:
        ...

    def close(self) -> None:
        ...

    def load_tetrahedra(self, xyz: Tensor, cells: Tensor, /) -> None:
        ...

    def find_visited_cells(self,
                           num_visited_cells: Tensor,
                           visited_cells: Tensor,
                           barycentric_coordinates: Tensor,
                           hit_distances: Tensor,
                           vertex_indices: Tensor,
                           distances: Tensor, /) -> _FindVisitedCellsResult:
        ...

    def find_tetrahedra(self, positions: Tensor, /) -> Tensor:
        ...

    def trace_rays(self,
                   ray_origins: Tensor,
                   ray_directions: Tensor,
                   max_ray_triangles: int) -> _TraceRaysResult:
        ...

    def trace_rays_triangles(self,
                             ray_origins: Tensor,
                             ray_directions: Tensor,
                             max_ray_triangles: int) -> _TraceRaysTrianglesResult:
        ...



def triangulate(points: Tensor, /) -> Tensor:
    ...


def gather_uint32(self: Tensor, dim: int, index: Tensor, /) -> Tensor:
    ...


def scatter_ema_uint32_(self: Tensor, dim: int, index: Tensor, decay: float, values: Tensor, /) -> None:
    ...


def add_barycentrics_grad(barycentrics: Tensor, vertices: Tensor, points: Tensor) -> Tensor:
    ...


def interpolate_values(vertex_indices: Tensor, barycentric_coordinates: Tensor, field: Tensor) -> Tensor:
    ...

import imp
from pathlib import Path

import torch

try:
    from . import tetranerf_cpp_extension as cpp
except ImportError as e:
    print("\033[91;1mERROR: Tetra-NeRF could not load the cpp extension. Build the project first.\033[0m")
    import_exception = e

    class LazyError:
        class LazyErrorObj:
            def __call__(self, *args, **kwds):
                raise RuntimeError(
                    "ERROR: Tetra-NeRF could not load cpp extension. Please build the project first"
                ) from import_exception

            def __getattribute__(self, __name: str):
                raise RuntimeError(
                    "ERROR: Tetra-NeRF could not load cpp extension. Please build the project first"
                ) from import_exception

        def __getattribute__(self, __name: str):
            return LazyError.LazyErrorObj()

    cpp = LazyError()

TetrahedraTracer = cpp.TetrahedraTracer
triangulate = cpp.triangulate


class _InterpolateValuesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertex_indices, barycentric_coordinates, field):
        output = cpp.interpolate_values(vertex_indices, barycentric_coordinates, field)
        ctx.save_for_backward(vertex_indices, barycentric_coordinates, field)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        vertex_indices, barycentric_coordinates, field = ctx.saved_tensors
        grad_field = cpp.interpolate_values_backward(
            vertex_indices, barycentric_coordinates, field, grad_out.contiguous()
        )
        return None, None, grad_field


def interpolate_values(vertex_indices, barycentric_coordinates, field):
    return _InterpolateValuesFunction.apply(vertex_indices, barycentric_coordinates, field)

import torch

try:
    from . import tetranerf_cpp_extension as cpp
except ImportError as err_:
    err = err_
    # TODO: Raise error
    print("\033[91;1mERROR: Tetra-NeRF could not load the cpp extension. Build the project first.\033[0m")

    class LazyError:
        class LazyErrorObj:
            def __call__(self, *args, **kwds):
                raise RuntimeError("ERROR: Tetra-NeRF could not load cpp extension. Please build the project first") from err

            def __getattribute__(self, __name: str):
                raise RuntimeError("ERROR: Tetra-NeRF could not load cpp extension. Please build the project first") from err

        def __getattribute__(self, __name: str):
            return LazyError.LazyErrorObj()

    cpp = LazyError()

TetrahedraTracer = cpp.TetrahedraTracer
triangulate = cpp.triangulate
gather_uint32 = cpp.gather_uint32
scatter_ema_uint32_ = cpp.scatter_ema_uint32


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


class _BarycentricsGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, barycentrics, vertices, points):
        ctx.save_for_backward(barycentrics, vertices)
        return barycentrics

    @staticmethod
    def backward(ctx, grad_barycentrics):
        barycentrics, vertices, = ctx.saved_tensors
        grad_vertices = None
        grad_points = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            t_mat = (vertices[..., 1:, :] - vertices[..., :1, :])
            m_vec = torch.linalg.solve(t_mat, grad_barycentrics)
            full_barycentrics = torch.cat([1.0-barycentrics.sum(-1, keepdim=True), barycentrics], -1)
        if ctx.needs_input_grad[1]:
            grad_vertices = (full_barycentrics.unsqueeze(-1) * m_vec.unsqueeze(-2)).mul_(-1.0)
        if ctx.needs_input_grad[2]:
            grad_points = m_vec
        return grad_barycentrics, grad_vertices, grad_points


def add_barycentrics_grad(barycentrics, vertices, points):
    return _BarycentricsGradFunction.apply(barycentrics, vertices, points)



def interpolate_values(vertex_indices, barycentric_coordinates, field):
    return _InterpolateValuesFunction.apply(vertex_indices, barycentric_coordinates, field)

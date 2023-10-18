#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <string>

#include "tetrahedra_tracer.h"
#include "triangulation.h"
#include "utils/exception.h"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) \
    CHECK_INPUT(x);         \
    CHECK_DEVICE(x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

struct PyTetrahedraTracer {
   public:
    PyTetrahedraTracer(const torch::Device &device) : device(device) {
        if (!device.is_cuda()) {
            throw Exception("The device argument must be a CUDA device.");
        }
        tracer = std::make_unique<TetrahedraTracer>(device.index());
    }
    ~PyTetrahedraTracer() {
        tracer.reset();
        tetrahedra_cells.reset();
        tetrahedra_vertices.reset();
    }
    py::dict trace_rays(const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions,
                        const unsigned long max_ray_triangles) {
        // Test if max_ray_triangles is a power of 2
        if ((max_ray_triangles & (max_ray_triangles - 1)) != 0) {
            throw Exception("max_ray_triangles must be a power of 2.");
        }
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);
        const size_t num_rays = ray_origins.numel() / 3;

        const auto num_visited_cells = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kInt32));
        const auto visited_cells = torch::zeros({(long)num_rays, (long)max_ray_triangles}, torch::device(device).dtype(torch::kInt32));
        const auto barycentric_coordinates = torch::zeros({(long)num_rays, (long)max_ray_triangles, 2, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto hit_distances = torch::zeros({(long)num_rays, (long)max_ray_triangles, 2}, torch::device(device).dtype(torch::kFloat32));
        const auto vertex_indices = torch::zeros({(long)num_rays, (long)max_ray_triangles, 4}, torch::device(device).dtype(torch::kInt32));
        tracer->trace_rays(
            num_rays,
            max_ray_triangles,
            reinterpret_cast<float3 *>(ray_origins.data_ptr()),
            reinterpret_cast<float3 *>(ray_directions.data_ptr()),
            reinterpret_cast<unsigned int *>(num_visited_cells.data_ptr()),
            reinterpret_cast<unsigned int *>(visited_cells.data_ptr()),
            reinterpret_cast<float3 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<float2 *>(hit_distances.data_ptr()),
            reinterpret_cast<uint4 *>(vertex_indices.data_ptr()));

        return py::dict(
            "num_visited_cells"_a = num_visited_cells,
            "visited_cells"_a = visited_cells,
            "barycentric_coordinates"_a = barycentric_coordinates,
            "vertex_indices"_a = vertex_indices,
            "hit_distances"_a = hit_distances);

    }

    py::dict trace_rays_triangles(const torch::Tensor &ray_origins,
                                  const torch::Tensor &ray_directions,
                                  const unsigned long max_ray_triangles) {
        // Test if max_ray_triangles is a power of 2
        if ((max_ray_triangles & (max_ray_triangles - 1)) != 0) {
            throw Exception("max_ray_triangles must be a power of 2.");
        }
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);
        const size_t num_rays = ray_origins.numel() / 3;

        const auto num_visited_triangles = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kInt32));
        const auto visited_triangles = torch::zeros({(long)num_rays, (long)max_ray_triangles}, torch::device(device).dtype(torch::kInt32));
        const auto barycentric_coordinates = torch::zeros({(long)num_rays, (long)max_ray_triangles, 2}, torch::device(device).dtype(torch::kFloat32));
        const auto hit_distances = torch::zeros({(long)num_rays, (long)max_ray_triangles}, torch::device(device).dtype(torch::kFloat32));
        const auto vertex_indices = torch::zeros({(long)num_rays, (long)max_ray_triangles, 3}, torch::device(device).dtype(torch::kInt32));
        tracer->trace_rays_triangles(
            num_rays,
            max_ray_triangles,
            reinterpret_cast<float3 *>(ray_origins.data_ptr()),
            reinterpret_cast<float3 *>(ray_directions.data_ptr()),
            reinterpret_cast<unsigned int *>(num_visited_triangles.data_ptr()),
            reinterpret_cast<unsigned int *>(visited_triangles.data_ptr()),
            reinterpret_cast<float2 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<float *>(hit_distances.data_ptr()),
            reinterpret_cast<uint3 *>(vertex_indices.data_ptr()));

        return py::dict(
            "num_visited_triangles"_a = num_visited_triangles,
            "visited_triangles"_a = visited_triangles,
            "barycentric_coordinates"_a = barycentric_coordinates,
            "vertex_indices"_a = vertex_indices,
            "hit_distances"_a = hit_distances);

    }

    py::dict find_tetrahedra(const torch::Tensor &positions) {
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(positions);
        const size_t num_points = positions.numel() / 3;
        auto shape = positions.sizes().vec();

        const auto barycentric_coordinates = torch::zeros(shape, torch::device(device).dtype(torch::kFloat32));

        shape.pop_back();
        shape.push_back(4);
        const auto vertex_indices = torch::zeros(shape, torch::device(device).dtype(torch::kInt32));

        shape.pop_back();
        const auto tetrahedra = torch::zeros(shape, torch::device(device).dtype(torch::kInt32));

        tracer->find_tetrahedra(
            num_points,
            reinterpret_cast<float3 *>(positions.data_ptr()),
            reinterpret_cast<unsigned int *>(tetrahedra.data_ptr()),
            reinterpret_cast<float3 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<uint4 *>(vertex_indices.data_ptr()));

        return py::dict(
            "tetrahedra"_a = tetrahedra,
            "barycentric_coordinates"_a = barycentric_coordinates,
            "vertex_indices"_a = vertex_indices,
            "valid_mask"_a = tetrahedra != -1);
    }

    void load_tetrahedra(
        const torch::Tensor &xyz,
        const torch::Tensor &cells) {
        CHECK_FLOAT_DIM3(xyz);
        CHECK_INPUT(cells);
        CHECK_DEVICE(cells);
        TORCH_CHECK(cells.size(-1) == 4, "indices must have last dimension with size 4")
        TORCH_CHECK(cells.dtype() == torch::kInt32, "indices must have int32 type")
        // TODO: WARNING: tetrahedra tracer expects uint32
        // We have to check if we are not overflowing 2,147,483,647
        this->tetrahedra_cells = cells;
        this->tetrahedra_vertices = xyz;
        tracer->load_tetrahedra(
            tetrahedra_vertices.value().numel() / 3,
            tetrahedra_cells.value().numel() / 4,
            reinterpret_cast<float3 *>(tetrahedra_vertices.value().data_ptr()),
            reinterpret_cast<uint4 *>(tetrahedra_cells.value().data_ptr()));
    }

    py::dict find_visited_cells(const torch::Tensor &num_visited_cells,
                                const torch::Tensor &visited_cells,
                                const torch::Tensor &barycentric_coordinates,
                                const torch::Tensor &hit_distances,
                                const torch::Tensor &vertex_indices,
                                const torch::Tensor &distances) {
        CHECK_INPUT(num_visited_cells);
        CHECK_DEVICE(num_visited_cells);
        CHECK_INPUT(visited_cells);
        CHECK_DEVICE(visited_cells);
        CHECK_INPUT(barycentric_coordinates);
        CHECK_DEVICE(barycentric_coordinates);
        CHECK_INPUT(hit_distances);
        CHECK_DEVICE(hit_distances);
        CHECK_INPUT(distances);
        CHECK_DEVICE(distances);
        CHECK_INPUT(vertex_indices);
        CHECK_DEVICE(vertex_indices);
        TORCH_CHECK(distances.dtype() == torch::kFloat32, "distances must have float32 type")
        const size_t num_rays = num_visited_cells.size(0);
        TORCH_CHECK(distances.dim() == 2 && distances.size(0) == num_rays, "distances must be of [num_rays, num_samples_per_ray] shape")
        TORCH_CHECK(vertex_indices.size(-1) == 4, "vertex_indices must have last dimension with size 4")
        const size_t num_samples_per_ray = distances.size(-1);
        const size_t num_vertices = tetrahedra_vertices.value().size(0);

        const auto mask = torch::full({(long)num_rays, (long)num_samples_per_ray}, 0, torch::device(device).dtype(torch::kBool));
        const auto matched_cells = torch::full({(long)num_rays, (long)num_samples_per_ray}, -1, torch::device(device).dtype(torch::kInt32));
        const auto barycentric_coordinates_out = torch::zeros({(long)num_rays, (long)num_samples_per_ray, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto vertex_indices_out = torch::full({(long)num_rays, (long)num_samples_per_ray, 4}, -1, torch::device(device).dtype(torch::kInt32));

        // Launch kernel to fill the interpolations
        // For now we will to this in pytorch
        find_matched_cells(
            num_rays,
            num_samples_per_ray,
            visited_cells.size(1),  // max_visited_cells
            reinterpret_cast<uint4 *>(tetrahedra_cells.value().data_ptr()),
            reinterpret_cast<unsigned int *>(num_visited_cells.data_ptr()),
            reinterpret_cast<unsigned int *>(visited_cells.data_ptr()),
            reinterpret_cast<float2 *>(hit_distances.data_ptr()),
            reinterpret_cast<float3 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<float *>(distances.data_ptr()),
            reinterpret_cast<uint4 *>(vertex_indices.data_ptr()),
            reinterpret_cast<unsigned int *>(matched_cells.data_ptr()),
            reinterpret_cast<uint4 *>(vertex_indices_out.data_ptr()),
            reinterpret_cast<bool *>(mask.data_ptr()),
            reinterpret_cast<float3 *>(barycentric_coordinates_out.data_ptr()));

        return py::dict(
            "cell_indices"_a = matched_cells,
            "vertex_indices"_a = vertex_indices_out,
            "mask"_a = mask,
            "barycentric_coordinates"_a = barycentric_coordinates_out);
    }

    const torch::Device &get_device() const {
        return this->device;
    }

   private:
    std::unique_ptr<TetrahedraTracer> tracer;
    torch::Device device;
    std::optional<torch::Tensor> tetrahedra_vertices;
    std::optional<torch::Tensor> tetrahedra_cells;
};

float py_find_average_spacing(const torch::Tensor &points) {
    CHECK_CONTIGUOUS(points);
    TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape [num_points, 3]");

    return find_average_spacing(
        points.size(0),
        reinterpret_cast<float3 *>(points.data_ptr()));
};

torch::Tensor py_triangulate(const torch::Tensor &points) {
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape [num_points, 3]");
    const auto points_ = points.cpu().contiguous();

    std::vector<uint4> cells = triangulate(
        points_.size(0),
        reinterpret_cast<float3 *>(points_.data_ptr()));

    if (cells.size() >= (size_t)std::numeric_limits<int>::max) {
        throw Exception("Too many points!");
    }
    auto cells_out = torch::empty({(long)cells.size(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    memcpy(
        cells_out.data_ptr(),
        reinterpret_cast<void *>(cells.data()),
        cells.size() * sizeof(uint4));
    return cells_out.to(points.device());
};

template <typename... Types>
inline void call_dynamic_interpolate_values(uint32_t interpolation_dim, Types... args) {
    switch (interpolation_dim) {
        case 2:
            interpolate_values<2>(args...);
            break;
        case 3:
            interpolate_values<3>(args...);
            break;
        case 4:
            interpolate_values<4>(args...);
            break;
        case 6:
            interpolate_values<6>(args...);
            break;
        default:
            throw Exception(("Unsupported interpolation dimension with value " + std::to_string(interpolation_dim)).c_str());
    }
}

template <typename... Types>
inline void call_dynamic_interpolate_values_backward(uint32_t interpolation_dim, Types... args) {
    switch (interpolation_dim) {
        case 2:
            interpolate_values_backward<2>(args...);
            break;
        case 3:
            interpolate_values_backward<3>(args...);
            break;
        case 4:
            interpolate_values_backward<4>(args...);
            break;
        case 6:
            interpolate_values_backward<6>(args...);
            break;
        default:
            throw Exception(("Unsupported interpolation dimension with value " + std::to_string(interpolation_dim)).c_str());
    }
}

torch::Tensor py_interpolate_values(const torch::Tensor &vertex_indices,
                                    const torch::Tensor &barycentric_coordinates,
                                    const torch::Tensor &field) {
    CHECK_INPUT(vertex_indices);
    CHECK_INPUT(barycentric_coordinates);
    CHECK_INPUT(field);
    TORCH_CHECK(vertex_indices.dtype() == torch::kInt32, "vertex_indices must be a tensor of type int32");
    TORCH_CHECK(barycentric_coordinates.dtype() == torch::kFloat32, "barycentric_coordinates must be a tensor of type float32");
    TORCH_CHECK(barycentric_coordinates.size(-1) + 1 == vertex_indices.size(-1), "barycentric_coordinates must have the same last dimension as vertex_indices - 1");
    TORCH_CHECK(field.dtype() == torch::kFloat32, "field must be a tensor of type float32");

    // if (vertex_indices.max().item<int>() >= field.size(-1)) {
    //     throw Exception("vertex_indices contains an index that is out of bounds for the field");
    // }

    const uint32_t interpolation_dim = vertex_indices.size(-1);
    const uint32_t num_values = vertex_indices.numel() / interpolation_dim;
    const uint32_t field_dim = field.size(0);
    const uint32_t num_vertices = field.size(-1);
    auto result_shape = vertex_indices.sizes().vec();
    result_shape.pop_back();
    result_shape.insert(result_shape.begin(), field_dim);
    const auto result = torch::empty(result_shape, torch::device(field.device()).dtype(field.dtype()));
    call_dynamic_interpolate_values(
        interpolation_dim,
        num_vertices,
        num_values,
        field_dim,
        reinterpret_cast<uint *>(vertex_indices.data_ptr()),
        reinterpret_cast<float *>(barycentric_coordinates.data_ptr()),
        reinterpret_cast<float *>(field.data_ptr()),
        reinterpret_cast<float *>(result.data_ptr()));
    return result.moveaxis(0, -1);
    const auto points = field.index({vertex_indices.to(torch::kLong)});
    // std::cout << points.sizes() << std::endl;
    // std::cout << vertex_indices.sizes() << std::endl;

    // Mask is not needed here
    // The values were masked already in barycentric coordinates
    // barycentric_coordinates.mul_(mask);
    return torch::einsum("rbij,rbi->rbj", {points, barycentric_coordinates});
}

torch::Tensor py_interpolate_values_backward(const torch::Tensor &vertex_indices,
                                             const torch::Tensor &barycentric_coordinates,
                                             const torch::Tensor &field,
                                             const torch::Tensor &grad_in) {
    CHECK_INPUT(vertex_indices);
    CHECK_INPUT(barycentric_coordinates);
    CHECK_INPUT(field);
    CHECK_INPUT(grad_in);
    TORCH_CHECK(vertex_indices.dtype() == torch::kInt32, "vertex_indices must be a tensor of type int32");
    TORCH_CHECK(barycentric_coordinates.dtype() == torch::kFloat32, "barycentric_coordinates must be a tensor of type float32");
    TORCH_CHECK(field.dtype() == torch::kFloat32, "field must be a tensor of type float32");
    TORCH_CHECK(grad_in.dtype() == torch::kFloat32, "grad_in must be a tensor of type float32");
    TORCH_CHECK(barycentric_coordinates.size(-1) + 1 == vertex_indices.size(-1), "barycentric_coordinates must have the same last dimension as vertex_indices - 1");
    // TODO: implement grad computation wrt barycentric coords for pose optimisation
    uint32_t interpolation_dim = vertex_indices.size(-1);
    const uint32_t num_values = vertex_indices.numel() / interpolation_dim;
    const uint32_t field_dim = field.size(0);
    const uint32_t num_vertices = field.size(-1);
    TORCH_CHECK(grad_in.size(-1) == field_dim, "grad_in must have shape [..., field_dim]");
    const auto grad_field_out = torch::zeros({(long)field_dim, (long)num_vertices}, torch::device(grad_in.device()).dtype(grad_in.dtype()));

    call_dynamic_interpolate_values_backward(
        interpolation_dim,
        num_vertices,
        num_values,
        field_dim,
        reinterpret_cast<uint *>(vertex_indices.data_ptr()),
        reinterpret_cast<float *>(barycentric_coordinates.data_ptr()),
        reinterpret_cast<float *>(grad_in.moveaxis(-1, 0).contiguous().data_ptr()),
        reinterpret_cast<float *>(grad_field_out.data_ptr()));
    return grad_field_out;
}

torch::Tensor py_gather_uint32(const torch::Tensor &self, const uint32_t dim, const torch::Tensor &index) {
    TORCH_CHECK(index.dtype() == torch::kInt32, "index must be a tensor of type int32");
    TORCH_CHECK(self.is_floating_point(), "self must be a tensor of a floating-point type");
    TORCH_CHECK(self.device() == index.device(), "self and index must be on the same device");
    TORCH_CHECK(self.is_contiguous(), "self must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(self.is_cuda(), "self must be on CUDA");
    TORCH_CHECK(index.is_cuda(), "index must be on CUDA");
    TORCH_CHECK(self.dim() == 1, "self must be 1-dimensional");
    TORCH_CHECK(index.dim() == self.dim(), "self and index must have the same number of dimensions");
    TORCH_CHECK(dim == 0, "dim must be 0");

    const auto result_shape = index.sizes();
    auto result = torch::empty(result_shape, torch::device(self.device()).dtype(self.dtype()));
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "gather_uint32", [&] {
        gather_uint32<scalar_t>(
            (uint32_t)self.numel(),
            (uint32_t)index.numel(),
            reinterpret_cast<uint32_t *>(index.data_ptr()),
            self.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>());
    });
    return result;
}


// TODO: newer pytorch now supports torch::Scalar pybind11 casting
// It is not, however available in older versions
// https://github.com/pytorch/pytorch/pull/90624
void py_scatter_ema_uint32(const torch::Tensor &self, const uint32_t dim, const torch::Tensor &index, const double decay, const torch::Tensor &values) {
    TORCH_CHECK(dim == 0, "dim must be 0");
    TORCH_CHECK(self.is_floating_point(), "self must be a tensor of a floating-point type");
    TORCH_CHECK(self.is_contiguous(), "self must be contiguous");
    TORCH_CHECK(self.dim() == 1, "self must be 1-dimensional");
    TORCH_CHECK(self.is_cuda(), "self must be on CUDA");

    TORCH_CHECK(index.dtype() == torch::kInt32, "index must be a tensor of type int32");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(index.is_cuda(), "index must be on CUDA");

    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");

    TORCH_CHECK(self.device() == index.device(), "self and index must be on the same device");
    TORCH_CHECK(values.device() == index.device(), "values and index must be on the same device");
    TORCH_CHECK(values.dtype() == self.dtype(), "values and self must have the same dtype");
    TORCH_CHECK(index.dim() == self.dim(), "self and index must have the same number of dimensions");
    TORCH_CHECK(values.sizes() == index.sizes(), "values and index must have the same shape");

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "scatter_ema_uint32", [&] {
        scatter_ema_uint32<scalar_t>(
            (uint32_t)self.numel(),
            (uint32_t)index.numel(),
            reinterpret_cast<uint32_t *>(index.data_ptr()),
            decay, // decay.to<scalar_t>(),
            values.data_ptr<scalar_t>(),
            self.data_ptr<scalar_t>());
    });
}

PYBIND11_MODULE(tetranerf_cpp_extension, m) {
    py::class_<PyTetrahedraTracer>(m, "TetrahedraTracer")
        .def(py::init<const torch::Device &>())
        .def_property_readonly("device", &PyTetrahedraTracer::get_device)
        .def("trace_rays", &PyTetrahedraTracer::trace_rays)
        .def("trace_rays_triangles", &PyTetrahedraTracer::trace_rays_triangles)
        .def("find_visited_cells", &PyTetrahedraTracer::find_visited_cells)
        .def("find_tetrahedra", &PyTetrahedraTracer::find_tetrahedra)
        .def("load_tetrahedra", &PyTetrahedraTracer::load_tetrahedra);

    m.def("triangulate", &py_triangulate);
    m.def("find_average_spacing", &py_find_average_spacing);
    m.def("interpolate_values", &py_interpolate_values);
    m.def("interpolate_values_backward", &py_interpolate_values_backward);
    m.def("gather_uint32", &py_gather_uint32);
    m.def("scatter_ema_uint32", &py_scatter_ema_uint32);
}
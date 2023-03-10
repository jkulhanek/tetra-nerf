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
                        const unsigned long max_visited_cells) {
        unsigned int max_ray_triangles = 2;
        while (max_ray_triangles < max_visited_cells*2) {
            max_ray_triangles <<= 1;
        }
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);
        const size_t num_rays = ray_origins.numel() / 3;

        const auto num_visited_cells = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kInt32));
        const auto visited_cells = torch::zeros({(long)num_rays, (long)max_visited_cells}, torch::device(device).dtype(torch::kInt32));
        const auto barycentric_coordinates = torch::zeros({(long)num_rays, (long)max_visited_cells, 2, 4}, torch::device(device).dtype(torch::kFloat32));
        const auto hit_distances = torch::zeros({(long)num_rays, (long)max_visited_cells, 2}, torch::device(device).dtype(torch::kFloat32));
        tracer->trace_rays(
            num_rays,
            max_ray_triangles,
            max_visited_cells,
            reinterpret_cast<float3 *>(ray_origins.data_ptr()),
            reinterpret_cast<float3 *>(ray_directions.data_ptr()),
            reinterpret_cast<unsigned int *>(num_visited_cells.data_ptr()),
            reinterpret_cast<unsigned int *>(visited_cells.data_ptr()),
            reinterpret_cast<float4 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<float *>(hit_distances.data_ptr()));

        return py::dict(
            "num_visited_cells"_a = num_visited_cells,
            "visited_cells"_a = visited_cells,
            "barycentric_coordinates"_a = barycentric_coordinates,
            "hit_distances"_a = hit_distances);
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
        TORCH_CHECK(distances.dtype() == torch::kFloat32, "distances must have float32 type")
        const size_t num_rays = num_visited_cells.size(0);
        TORCH_CHECK(distances.dim() == 2 && distances.size(0) == num_rays, "distances must be of [num_rays, num_samples_per_ray] shape")
        const size_t num_samples_per_ray = distances.size(-1);
        const size_t num_vertices = tetrahedra_vertices.value().size(0);

        const auto mask = torch::full({(long)num_rays, (long)num_samples_per_ray}, 0, torch::device(device).dtype(torch::kBool));
        const auto matched_cells = torch::full({(long)num_rays, (long)num_samples_per_ray}, -1, torch::device(device).dtype(torch::kInt32));
        const auto barycentric_coordinates_out = torch::zeros({(long)num_rays, (long)num_samples_per_ray, 4}, torch::device(device).dtype(torch::kFloat32));
        const auto vertex_indices = torch::zeros({(long)num_rays, (long)num_samples_per_ray, 4}, torch::device(device).dtype(torch::kInt32));

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
            reinterpret_cast<float4 *>(barycentric_coordinates.data_ptr()),
            reinterpret_cast<float *>(distances.data_ptr()),
            reinterpret_cast<unsigned int *>(matched_cells.data_ptr()),
            reinterpret_cast<uint4 *>(vertex_indices.data_ptr()),
            reinterpret_cast<bool *>(mask.data_ptr()),
            reinterpret_cast<float4 *>(barycentric_coordinates_out.data_ptr()));

        return py::dict(
            "cell_indices"_a = matched_cells,
            "vertex_indices"_a = vertex_indices,
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
    CHECK_CONTIGUOUS(points);
    TORCH_CHECK(points.device().is_cpu(), "points must be a CPU tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape [num_points, 3]");

    std::vector<uint4> cells = triangulate(
        points.size(0),
        reinterpret_cast<float3 *>(points.data_ptr()));

    if (cells.size() >= (size_t)std::numeric_limits<int>::max) {
        throw Exception("Too many points!");
    }
    auto cells_out = torch::empty({(long)cells.size(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    memcpy(
        cells_out.data_ptr(),
        reinterpret_cast<void *>(cells.data()),
        cells.size() * sizeof(uint4));
    return cells_out;
};

torch::Tensor py_interpolate_values(const torch::Tensor &vertex_indices,
                                    const torch::Tensor &barycentric_coordinates,
                                    const torch::Tensor &field) {
    CHECK_INPUT(vertex_indices);
    CHECK_INPUT(barycentric_coordinates);
    CHECK_INPUT(field);
    const size_t num_values = vertex_indices.numel() / 4;
    const size_t field_dim = field.size(-1);
    const auto result = torch::empty({barycentric_coordinates.size(0), barycentric_coordinates.size(1), field.size(-1)}, torch::device(field.device()).dtype(field.dtype()));
    interpolate_values(
        num_values,
        field_dim,
        reinterpret_cast<uint4 *>(vertex_indices.data_ptr()),
        reinterpret_cast<float4 *>(barycentric_coordinates.data_ptr()),
        reinterpret_cast<float *>(field.data_ptr()),
        reinterpret_cast<float *>(result.data_ptr()));
    return result;
    const auto points = field.index({vertex_indices.to(torch::kLong)});
    std::cout << points.sizes() << std::endl;
    std::cout << vertex_indices.sizes() << std::endl;

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
    // TODO: implement grad computation wrt barycentric coords for pose optimisation
    const size_t num_values = vertex_indices.numel() / 4;
    const size_t field_dim = field.size(-1);
    const size_t num_vertices = field.size(0);
    const auto grad_field_out = torch::zeros({(long)num_vertices, (long)field_dim}, torch::device(grad_in.device()).dtype(grad_in.dtype()));
    interpolate_values_backward(
        num_vertices,
        num_values,
        field_dim,
        reinterpret_cast<uint4 *>(vertex_indices.data_ptr()),
        reinterpret_cast<float4 *>(barycentric_coordinates.data_ptr()),
        reinterpret_cast<float *>(grad_in.data_ptr()),
        reinterpret_cast<float *>(grad_field_out.data_ptr()));
    return grad_field_out;
}

    
PYBIND11_MODULE(tetranerf_cpp_extension, m) {
    py::class_<PyTetrahedraTracer>(m, "TetrahedraTracer")
        .def(py::init<const torch::Device &>())
        .def_property_readonly("device", &PyTetrahedraTracer::get_device)
        .def("trace_rays", &PyTetrahedraTracer::trace_rays)
        .def("find_visited_cells", &PyTetrahedraTracer::find_visited_cells)
        .def("load_tetrahedra", &PyTetrahedraTracer::load_tetrahedra);

    m.def("triangulate", &py_triangulate);
    m.def("find_average_spacing", &py_find_average_spacing);
    m.def("interpolate_values", &py_interpolate_values);
    m.def("interpolate_values_backward", &py_interpolate_values_backward);
}
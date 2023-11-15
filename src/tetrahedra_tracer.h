#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern unsigned char ptx_code_file[];
extern unsigned char ptx_code_file_find_tetrahedra[];
extern unsigned char ptx_code_file_triangles[];

class TetrahedraStructure {
   public:
    TetrahedraStructure() noexcept;
    TetrahedraStructure(const OptixDeviceContext &context, const uint8_t device) : device(device), context(context) {}
    TetrahedraStructure(
        const OptixDeviceContext &context,
        const uint8_t device,
        const size_t num_vertices,
        const size_t num_cells,
        const float3 *d_vertices,
        const uint4 *cells) : TetrahedraStructure(context, device) {
        build(num_vertices, num_cells, d_vertices, cells);
    }

    ~TetrahedraStructure() noexcept(false);
    TetrahedraStructure(const TetrahedraStructure &) = delete;
    TetrahedraStructure &operator=(const TetrahedraStructure &) = delete;
    TetrahedraStructure(TetrahedraStructure &&other) noexcept;
    TetrahedraStructure &operator=(TetrahedraStructure &&other) {
        using std::swap;
        if (this != &other) {
            TetrahedraStructure tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    friend void swap(TetrahedraStructure &first, TetrahedraStructure &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.num_vertices, second.num_vertices);
        swap(first.num_cells, second.num_cells);
        swap(first.gas_handle_, second.gas_handle_);
        swap(first.d_gas_output_buffer, second.d_gas_output_buffer);
        swap(first.tetrahedra_vertices, second.tetrahedra_vertices);
        swap(first.triangle_indices_, second.triangle_indices_);
        swap(first.triangle_tetrahedra_, second.triangle_tetrahedra_);
    }

    OptixTraversableHandle gas_handle() const {
        if (!defined()) {
            throw std::runtime_error("TetrahedraStructure is not initialized");
        }
        return gas_handle_;
    }

    const uint3 *triangle_indices() const {
        return triangle_indices_;
    }

    const uint2 *triangle_tetrahedra() const {
        return triangle_tetrahedra_;
    }

    bool defined() const {
        return gas_handle_ != 0;
    }

   private:
    void build(
        const size_t num_vertices,
        const size_t num_cells,
        const float3 *d_vertices,
        const uint4 *cells);

    void release();
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    size_t num_vertices = 0;
    size_t num_cells = 0;
    OptixTraversableHandle gas_handle_ = 0;
    CUdeviceptr d_gas_output_buffer = 0;
    const float3 *tetrahedra_vertices = nullptr;
    uint3 *triangle_indices_ = nullptr;
    uint2 *triangle_tetrahedra_ = nullptr;
};

class TraceRaysPipeline {
   public:
    TraceRaysPipeline() = default;
    TraceRaysPipeline(const OptixDeviceContext &context, int8_t device);
    TraceRaysPipeline(const TraceRaysPipeline &) = delete;
    TraceRaysPipeline &operator=(const TraceRaysPipeline &) = delete;
    TraceRaysPipeline(TraceRaysPipeline &&other) noexcept;
    TraceRaysPipeline &operator=(TraceRaysPipeline &&other) {
        using std::swap;
        if (this != &other) {
            TraceRaysPipeline tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~TraceRaysPipeline() noexcept(false);

    friend void swap(TraceRaysPipeline &first, TraceRaysPipeline &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
        swap(first.eps, second.eps);
    }

    void trace_rays(
        const TetrahedraStructure *tetrahedra_structure,
        const size_t num_rays,
        const unsigned int max_ray_triangles,
        const float3 *ray_origins,
        const float3 *ray_directions,
        unsigned int *num_visited_cells_out,
        unsigned int *visited_cells_out,
        float3 *barycentric_coordinates_out,
        float2 *hit_distances_out,
        uint4 *vertex_indices_out);

   private:
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    float eps = 1e-6;

    static std::string load_ptx_data() {
        return std::string((char *)ptx_code_file);
    }
};

class TraceRaysTrianglesPipeline {
   public:
    TraceRaysTrianglesPipeline() = default;
    TraceRaysTrianglesPipeline(const OptixDeviceContext &context, int8_t device);
    TraceRaysTrianglesPipeline(const TraceRaysTrianglesPipeline &) = delete;
    TraceRaysTrianglesPipeline &operator=(const TraceRaysTrianglesPipeline &) = delete;
    TraceRaysTrianglesPipeline(TraceRaysTrianglesPipeline &&other) noexcept;
    TraceRaysTrianglesPipeline &operator=(TraceRaysTrianglesPipeline &&other) {
        using std::swap;
        if (this != &other) {
            TraceRaysTrianglesPipeline tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~TraceRaysTrianglesPipeline() noexcept(false);
    friend void swap(TraceRaysTrianglesPipeline &first, TraceRaysTrianglesPipeline &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
    }

    void trace_rays(
        const TetrahedraStructure *tetrahedra_structure,
        const size_t num_rays,
        const unsigned int max_ray_triangles,
        const float3 *ray_origins,
        const float3 *ray_directions,
        unsigned int *num_visited_triangles_out,
        unsigned int *visited_triangles_out,
        float2 *barycentric_coordinates_out,
        float *hit_distances_out,
        uint3 *vertex_indices_out);

   private:
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    static std::string load_ptx_data() {
        return std::string((char *)ptx_code_file_triangles);
    }
};


class FindTetrahedraPipeline {
   public:
    FindTetrahedraPipeline() = default;
    FindTetrahedraPipeline(const OptixDeviceContext &context, int8_t device);
    FindTetrahedraPipeline(const FindTetrahedraPipeline &) = delete;
    FindTetrahedraPipeline &operator=(const FindTetrahedraPipeline &) = delete;
    FindTetrahedraPipeline(FindTetrahedraPipeline &&other) noexcept;
    FindTetrahedraPipeline &operator=(FindTetrahedraPipeline &&other) {
        using std::swap;
        if(this != &other) {
            FindTetrahedraPipeline tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~FindTetrahedraPipeline() noexcept(false);
    friend void swap(FindTetrahedraPipeline &first, FindTetrahedraPipeline &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
    }

    void find_tetrahedra(const TetrahedraStructure *tetrahedra_structure,
                         const size_t num_points,
                         const float3 *points,
                         unsigned int *tetrahedra_out,
                         float3 *barycentric_coordinates_out,
                         uint4 *vertex_indices_out);

   private:
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    float eps = 1e-6;

    static std::string load_ptx_data() {
        return std::string((char *)ptx_code_file_find_tetrahedra);
    }
};

class TetrahedraTracer {
   public:
    TetrahedraTracer(int8_t device);
    ~TetrahedraTracer() noexcept(false);
    TetrahedraTracer(const TetrahedraTracer &) = delete;
    TetrahedraTracer &operator=(const TetrahedraTracer &) = delete;
    TetrahedraTracer &operator=(TetrahedraTracer &&other) {
        using std::swap;
        if (this != &other) {
            TetrahedraTracer tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    TetrahedraTracer(TetrahedraTracer &&other);

    // Loads the tetrahedra and builds an accel structure
    // NOTE: the d_vertices and cells arrays are not copied!
    // They must be valid for the existence of TetrahydraTracer
    // We do this to save some memory
    void load_tetrahedra(const size_t num_vertices,
                         const size_t num_cells,
                         const float3 *d_vertices,
                         const uint4 *cells) {
        tetrahedra_structure = std::move(TetrahedraStructure(context, device, num_vertices, num_cells, d_vertices, cells));
    }

    void trace_rays(const size_t num_rays,
                    const unsigned int max_ray_triangles,
                    const float3 *ray_origins,
                    const float3 *ray_directions,
                    unsigned int *num_visited_cells_out,
                    unsigned int *visited_cells_out,
                    float3 *barycentric_coordinates_out,
                    float2 *hit_distances_out,
                    uint4 *vertex_indices_out) {
        trace_rays_pipeline.trace_rays(
            &tetrahedra_structure,
            num_rays,
            max_ray_triangles,
            ray_origins,
            ray_directions,
            num_visited_cells_out,
            visited_cells_out,
            barycentric_coordinates_out,
            hit_distances_out,
            vertex_indices_out);
    }

    void trace_rays_triangles(const size_t num_rays,
                    const unsigned int max_ray_triangles,
                    const float3 *ray_origins,
                    const float3 *ray_directions,
                    unsigned int *num_visited_triangles_out,
                    unsigned int *visited_triangles_out,
                    float2 *barycentric_coordinates_out,
                    float *hit_distances_out,
                    uint3 *vertex_indices_out) {
        trace_rays_triangles_pipeline.trace_rays(
            &tetrahedra_structure,
            num_rays,
            max_ray_triangles,
            ray_origins,
            ray_directions,
            num_visited_triangles_out,
            visited_triangles_out,
            barycentric_coordinates_out,
            hit_distances_out,
            vertex_indices_out);
    }

    void find_tetrahedra(const size_t num_points,
                         const float3 *points,
                         unsigned int *tetrahedra_out,
                         float3 *barycentric_coordinates_out,
                         uint4 *vertex_indices_out) {
        find_tetrahedra_pipeline.find_tetrahedra(
            &tetrahedra_structure,
            num_points,
            points,
            tetrahedra_out,
            barycentric_coordinates_out,
            vertex_indices_out);
    }

   private:
    // Global contexts
    int8_t device;
    OptixDeviceContext context;

    TetrahedraStructure tetrahedra_structure;
    TraceRaysPipeline trace_rays_pipeline;
    TraceRaysTrianglesPipeline trace_rays_triangles_pipeline;
    FindTetrahedraPipeline find_tetrahedra_pipeline;
};

void find_matched_cells(const size_t num_rays,
                        const size_t num_samples_per_ray,
                        const size_t max_visited_cells,
                        const uint4 *cells,
                        const unsigned int *num_visited_cells,
                        const unsigned int *visited_cells,
                        const float2 *hit_distances,
                        const float3 *barycentric_coordinates,
                        const float *distances,
                        const uint4 *vertex_indices,
                        unsigned int *matched_cells_out,
                        uint4 *matched_vertices_out,
                        bool *mask_out,
                        float3 *barycentric_coordinates_out);

template <uint32_t interpolation_dim>
void interpolate_values(
    const uint32_t num_vertices,
    const uint32_t num_values,
    const uint32_t field_dim,
    const uint32_t *vertex_indices,
    const float *barycentric_coordinates,
    const float *field,
    float *result);
template <uint32_t interpolation_dim>
void interpolate_values_backward(const uint32_t num_vertices,
                                 const uint32_t num_values,
                                 const uint32_t field_dim,
                                 const uint32_t *vertex_indices,
                                 const float *barycentric_coordinates,
                                 const float *grad_in,
                                 float *field_grad_out);



template <typename scalar_t>
void gather_uint32(const uint32_t num_values, const uint32_t num_indices, const uint32_t *indices, const scalar_t *values, scalar_t *result);

template <typename scalar_t>
void scatter_ema_uint32(const uint32_t num_result, const uint32_t num_indices, const uint32_t *indices, const scalar_t decay, const scalar_t *values, scalar_t *result);
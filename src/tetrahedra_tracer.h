#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

extern unsigned char ptx_code_file[];

class TetrahedraTracer {
   public:
    TetrahedraTracer(int8_t device);
    ~TetrahedraTracer() noexcept(false);

    // Loads the tetrahedra and builds an accel structure
    // NOTE: the d_vertices and cells arrays are not copied!
    // They must be valid for the existence of TetrahydraTracer
    // We do this to save some memory
    void load_tetrahedra(const size_t num_vertices,
                         const size_t num_cells,
                         const float3 *d_vertices,
                         const uint4 *cells);

    void trace_rays(const size_t num_rays,
                    const unsigned int max_ray_triangles,
                    const unsigned int max_visited_cells,
                    const float3 *ray_origins,
                    const float3 *ray_directions,
                    unsigned int *num_visited_cells_out,
                    unsigned int *visited_cells_out,
                    float4 *barycentric_coordinates_out,
                    float *hit_distances_out);

   private:
    // Global contexts
    int8_t device;
    OptixDeviceContext context;
    OptixModule module;
    OptixShaderBindingTable sbt;
    OptixPipeline pipeline;

    CUstream stream;
    CUdeviceptr d_param;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    // Accel structure
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    const float3 *tetrahedra_vertices;
    const uint4 *tetrahedra_cells;
    size_t num_vertices;
    size_t num_cells;
    float eps = 1e-6;

    static std::string load_ptx_data();
    void release_accel_structure();
};

void convert_cells_to_vertices(const size_t block_size, const size_t n, const uint4 *cells, uint3 *triangle_indices);

void post_process_tetrahedra(const size_t num_rays,
                             const size_t max_visited_triangles,
                             const size_t max_visited_cells,
                             const unsigned int *num_visited_triangles,
                             unsigned int *visited_triangles,
                             const float2 *barycentric_coordinates_triangle,
                             const float *distances,
                             unsigned int *visited_cells,
                             unsigned int *num_visited_cells,
                             float4 *barycentric_coordinates,
                             float *cell_distances,
                             const float eps);

void find_matched_cells(const size_t num_rays,
                        const size_t num_samples_per_ray,
                        const size_t max_visited_cells,
                        const uint4 *cells,
                        const unsigned int *num_visited_cells,
                        const unsigned int *visited_cells,
                        const float2 *hit_distances,
                        const float4 *barycentric_coordinates,
                        const float *distances,
                        unsigned int *matched_cells_out,
                        uint4 *matched_vertices_out,
                        bool *mask_out,
                        float4 *barycentric_coordinates_out);

void interpolate_values(
    const size_t num_values,
    const size_t field_dim,
    const size_t num_vertices,
    const uint4 *vertex_indices,
    const float4 *barycentric_coordinates,
    const float *field,
    float *result);

void interpolate_values_backward(const size_t num_vertices,
                                 const size_t num_values,
                                 const size_t field_dim,
                                 const uint4 *vertex_indices,
                                 const float4 *barycentric_coordinates,
                                 const float *grad_in,
                                 float *field_grad_out);
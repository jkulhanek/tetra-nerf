#include <assert.h>

#include <iostream>
#include <limits>

#include "tetrahedra_tracer.h"

__global__ void cu_convert_cells_to_vertices(int n, const uint4 *cells, uint3 *vertices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        const uint4 cell = cells[i];
        vertices[i * 4 + 0] = make_uint3(cell.y, cell.z, cell.w);
        vertices[i * 4 + 1] = make_uint3(cell.z, cell.w, cell.x);
        vertices[i * 4 + 2] = make_uint3(cell.w, cell.x, cell.y);
        vertices[i * 4 + 3] = make_uint3(cell.x, cell.y, cell.z);
    }
}

void convert_cells_to_vertices(const size_t n, const uint4 *cells, uint3 *triangle_indices) {
    const size_t block_size = 1024;
    cu_convert_cells_to_vertices<<<(n + block_size - 1) / block_size, block_size>>>(n, cells, triangle_indices);
}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename float_type>
__global__ void gather_uint32_kernel(const uint32_t num_values,
                                     const uint32_t num_indices,
                                     const uint32_t *indices,
                                     const float_type *values,
                                     float_type *result) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;

    const auto target_ind = indices[i];
    if (target_ind >= num_values) return;
    result[i] = values[target_ind];
}

template <typename float_type>
void gather_uint32(const uint32_t num_values,
                   const uint32_t num_indices,
                   const uint32_t *indices,
                   const float_type *values,
                   float_type *result) {
    const uint32_t block_size = 1024;
    gather_uint32_kernel<float_type><<<div_round_up(num_indices, block_size), block_size>>>(num_values, num_indices, indices, values, result);
}


template <typename float_type>
__forceinline__ __device__ float_type atomicEMA(float_type* address, const float_type decay, const float_type update);

template <>
__forceinline__ __device__ float atomicEMA<float>(float* address, const float decay, const float update) {
    uint32_t* address_as_u = (uint32_t*)address;
    uint32_t old = *address_as_u, assumed;
    float new_val;
    do {
        assumed = old;
        new_val = __float_as_uint(__uint_as_float(assumed) * decay + (1 - decay) * update);
        old = atomicCAS(address_as_u, assumed, new_val);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return new_val;
}

template <>
__forceinline__ __device__ double atomicEMA<double>(double* address, const double decay, const double update) {
    unsigned long long int* address_as_u = (unsigned long long int*)address;
    unsigned long long int old = *address_as_u, assumed;
    float new_val;
    do {
        assumed = old;
        new_val = __double_as_longlong(__longlong_as_double(assumed) * decay + (1 - decay) * update);
        old = atomicCAS(address_as_u, assumed, new_val);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return new_val;
}

template <typename scalar_t>
__global__ void scatter_ema_uint32_kernel(const uint32_t num_result,
                                      const uint32_t num_indices,
                                      const uint32_t *indices,
                                      const scalar_t decay,
                                      const scalar_t *values,
                                      scalar_t *result) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;

    const auto target_ind = indices[i];
    if (target_ind >= num_result) return;

    atomicEMA(&result[target_ind], decay, values[i]);
}

template <typename float_type>
void scatter_ema_uint32(const uint32_t num_result,
                    const uint32_t num_indices,
                    const uint32_t *indices,
                    const float_type decay,
                    const float_type *values,
                    float_type *result) {
    const uint32_t block_size = 1024;
    scatter_ema_uint32_kernel<float_type><<<div_round_up(num_indices, block_size), block_size>>>(num_result, num_indices, indices, decay, values, result);
}

__global__ void find_matched_cells_kernel(const size_t num_rays,
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
                                          float3 *barycentric_coordinates_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < num_rays; i += stride) {
        unsigned int current_cell_pointer = 0;
        for (size_t j = 0; j < num_samples_per_ray; ++j) {
            const float current_distance = distances[i * num_samples_per_ray + j];
            while (current_cell_pointer < num_visited_cells[i] && hit_distances[i * max_visited_cells + current_cell_pointer].y < current_distance) current_cell_pointer++;
            if (current_cell_pointer >= num_visited_cells[i]) {
                // There will be no more matches on this ray
                break;
            }

            const float2 current_hit_distance = hit_distances[i * max_visited_cells + current_cell_pointer];
            if (current_hit_distance.x <= current_distance) {
                // We have a hit
                const unsigned int visited_cell = visited_cells[i * max_visited_cells + current_cell_pointer];
                mask_out[i * num_samples_per_ray + j] = 1;
                matched_cells_out[i * num_samples_per_ray + j] = visited_cell;
                matched_vertices_out[i * num_samples_per_ray + j] = vertex_indices[i * max_visited_cells + current_cell_pointer];

                float mult = (current_distance - current_hit_distance.x) / (current_hit_distance.y - current_hit_distance.x);
                const float3 coords1 = barycentric_coordinates[(i * max_visited_cells + current_cell_pointer) * 2];
                const float3 coords2 = barycentric_coordinates[(i * max_visited_cells + current_cell_pointer) * 2 + 1];
                barycentric_coordinates_out[i * num_samples_per_ray + j] = make_float3(
                    (1 - mult) * coords1.x + mult * coords2.x,
                    (1 - mult) * coords1.y + mult * coords2.y,
                    (1 - mult) * coords1.z + mult * coords2.z);
            }
            // Else there is no match - space between two cells
        }
    }
}

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
                        float3 *barycentric_coordinates_out) {
    const size_t block_size = 16;
    find_matched_cells_kernel<<<(num_rays + block_size - 1) / block_size, block_size>>>(
        num_rays,
        num_samples_per_ray,
        max_visited_cells,
        cells,
        num_visited_cells,
        visited_cells,
        hit_distances,
        barycentric_coordinates,
        distances,
        vertex_indices,
        matched_cells_out,
        matched_vertices_out,
        mask_out,
        barycentric_coordinates_out);
}

template <uint32_t interpolation_dim>
__global__ void interpolate_values_kernel(const uint32_t num_vertices,
                                          const uint32_t num_values,
                                          const uint32_t field_dim,
                                          const uint32_t *vertex_indices,
                                          const float *barycentric_coordinates,
                                          const float *field,
                                          float *result) {
    constexpr unsigned int empty = ~((unsigned int)0u);
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_values) return;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0;
    float weight = 0;
#pragma unroll
    for (uint32_t k = 0; k < interpolation_dim - 1; ++k) {
        const float &w = barycentric_coordinates[i * (interpolation_dim - 1) + k];
        const auto &vi = vertex_indices[i * interpolation_dim + k + 1];
        if (vi != empty)
            out += w * field[j * num_vertices + vi];
        weight += w;
    }
    if (vertex_indices[i * interpolation_dim] != empty)
        out += (1.0f - weight) * field[j * num_vertices + vertex_indices[i * interpolation_dim]];
    result[j * num_values + i] = out;
}

template <uint32_t interpolation_dim>
__global__ void interpolate_values_backward_kernel(const uint32_t num_vertices,
                                                   const uint32_t num_values,
                                                   const uint32_t field_dim,
                                                   const uint32_t *vertex_indices,
                                                   const float *barycentric_coordinates,
                                                   const float *grad_in,
                                                   float *field_grad_out) {
    constexpr unsigned int empty = ~((unsigned int)0u);
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_values) return;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    const auto lgradin = grad_in[j * num_values + i];
    float weight = 0;
#pragma unroll
    for (uint32_t k = 0; k < interpolation_dim - 1; ++k) {
        const float &w = barycentric_coordinates[i * (interpolation_dim - 1) + k];
        const auto &vi = vertex_indices[i * interpolation_dim + k + 1];
        if (vi != empty)
            atomicAdd(&field_grad_out[j * num_vertices + vi], w * lgradin);
        weight += w;
    }
    if (vertex_indices[i * interpolation_dim] != empty)
        atomicAdd(&field_grad_out[j * num_vertices + vertex_indices[i * interpolation_dim]], (1 - weight) * lgradin);
}

template <uint32_t interpolation_dim>
void interpolate_values(const uint32_t num_vertices,
                        const uint32_t num_values,
                        const uint32_t field_dim,
                        const uint32_t *vertex_indices,
                        const float *barycentric_coordinates,
                        const float *field,
                        float *result) {
    const uint32_t block_size = 1024;
    interpolate_values_kernel<interpolation_dim><<<dim3(div_round_up(num_values, block_size), field_dim), block_size>>>(
        num_vertices, num_values, field_dim, vertex_indices, barycentric_coordinates, field, result);
}

template void interpolate_values<2>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *field, float *result);
template void interpolate_values<3>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *field, float *result);
template void interpolate_values<4>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *field, float *result);
template void interpolate_values<6>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *field, float *result);

template <uint32_t interpolation_dim>
void interpolate_values_backward(const uint32_t num_vertices,
                                 const uint32_t num_values,
                                 const uint32_t field_dim,
                                 const uint32_t *vertex_indices,
                                 const float *barycentric_coordinates,
                                 const float *grad_in,
                                 float *field_grad_out) {
    const uint32_t block_size = 1024;
    interpolate_values_backward_kernel<interpolation_dim><<<dim3(div_round_up(num_values, block_size), field_dim), block_size>>>(
        num_vertices,
        num_values,
        field_dim,
        vertex_indices,
        barycentric_coordinates,
        grad_in,
        field_grad_out);
}

template void interpolate_values_backward<2>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *grad_in, float *field_grad_out);
template void interpolate_values_backward<3>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *grad_in, float *field_grad_out);
template void interpolate_values_backward<4>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *grad_in, float *field_grad_out);
template void interpolate_values_backward<6>(const uint32_t num_vertices, const uint32_t num_values, const uint32_t field_dim, const uint32_t *vertex_indices, const float *barycentric_coordinates, const float *grad_in, float *field_grad_out);

template void gather_uint32<float>(const uint32_t num_values, const uint32_t num_indices, const uint32_t *indices, const float *values, float *result);
template void gather_uint32<double>(const uint32_t num_values, const uint32_t num_indices, const uint32_t *indices, const double *values, double *result);
template void scatter_ema_uint32<float>(const uint32_t num_result, const uint32_t num_indices, const uint32_t *indices, const float decay, const float *values, float *result);
template void scatter_ema_uint32<double>(const uint32_t num_result, const uint32_t num_indices, const uint32_t *indices, const double decay, const double *values, double *result);
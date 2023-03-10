#include <assert.h>
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

void convert_cells_to_vertices(const size_t block_size, const size_t n, const uint4 *cells, uint3 *triangle_indices) {
    cu_convert_cells_to_vertices<<<(n + block_size - 1) / block_size, block_size>>>(n, cells, triangle_indices);
}

template <typename T>
__forceinline__ __device__ void swap(T &first, T &second) {
    const T mid = first;
    first = second;
    second = mid;
}

__forceinline__ __device__ float4 get_tetrahedra_barycentric_coordinates(const float2 triangle_coords, const unsigned int face) {
    assert(face < 4);
    const float b = triangle_coords.x;
    const float c = triangle_coords.y;
    const float a = 1.0f - b - c;
    if (face == 0) {
        return make_float4(0, a, b, c);
    } else if (face == 1) {
        return make_float4(c, 0, a, b);
    } else if (face == 2) {
        return make_float4(b, c, 0, a);
    } else if (face == 3) {
        return make_float4(a, b, c, 0);
    }
    return make_float4(0, 0, 0, 0);
}

__global__ void post_process_tetrahedra_kernel(const size_t num_rays,
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
                                               float eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < num_rays; i += stride) {
        const unsigned int ray_len = num_visited_triangles[i];
        size_t jc = 0;  // Local index to next cell to write

        // Dedupe triangles
        // TODO: do this more efficiently!!
        auto t = visited_triangles + (i * max_visited_triangles);
        const auto dl = distances + (i * max_visited_triangles);
        const auto empty = std::numeric_limits<unsigned int>::max();
        for (size_t j = 0; j + 1 < ray_len; ++j) {
            if (t[j] == empty) {
                continue;
            }
            float dn = dl[j];
            bool clear_self = false;
            for (size_t offset = 1; j + offset < ray_len && (t[j + offset] == empty || abs(dl[j + offset] - dn) < eps); offset++) {
                if (t[j + offset] != empty && t[j] / 4 == t[j + offset] / 4) {
                    // Triangle registered two times
                    // We only count it once, there can be exitting face later on
                    if (t[j] != t[j + offset]) {
                        // This is the case where the tetrahedra was hitted twice
                        // But the length of the intersection is too small to register
                        // We ignore the cell altogether
                        clear_self = true;
                    }
                    t[j + offset] = empty;
                }
            }
            if (clear_self) {
                t[j] = empty;
            }
        }

        for (size_t j = 0; j < ray_len; ++j) {
            if (t[j] == empty) {
                continue;
            }

            const auto cell = t[j] / 4;
            auto dn = dl[j];
            size_t real_offset = 1;
            bool was_error = true;
            for (size_t offset = 1;
                 j + offset < ray_len && (real_offset < 3 || t[j + offset] == empty || abs(dl[j + offset] - dn) < eps);
                 offset++) {
                if (t[j + offset] == empty) {
                    continue;
                }

                if (t[j + offset] / 4 == cell) {
                    // Close face - dump to output
                    const auto jc_g = i * max_visited_cells + jc;
                    visited_cells[jc_g] = t[j] / 4;
                    barycentric_coordinates[jc_g * 2] = get_tetrahedra_barycentric_coordinates(
                        barycentric_coordinates_triangle[i * max_visited_triangles + j],
                        t[j] % 4);
                    barycentric_coordinates[jc_g * 2 + 1] = get_tetrahedra_barycentric_coordinates(
                        barycentric_coordinates_triangle[i * max_visited_triangles + j + offset],
                        t[j + offset] % 4);
                    cell_distances[jc_g * 2] = dl[j];
                    cell_distances[jc_g * 2 + 1] = dl[j + offset];
                    jc++;

                    // If the closing triangle is the next one, move pointer, otherwise, mask the triangle to skip it during traversal
                    // Rotate next faces to
                    if (offset == 1) {
                        j += 1;
                    } else {
                        t[j + offset] = empty;
                    }
                    was_error = false;
                    break;
                }
                dn = dl[j + offset];
                real_offset++;
            }

            if (was_error && real_offset > 1) {
                // If real offest == 1, this is the last triangle on the ray
                // The ray might have been terminated prematurely because
                // the max_visited_triangles wasn't large enough
                // Therefore, this should not be considered an error.
                // if (i == 97810) {
                // printf("error on ray: %d\n", i);
                // for (size_t j = 0; j < jc; ++j) {
                //     printf("%05d \n", visited_cells[i*max_visited_cells+j]);
                // }
                // }
                // There was an error, we could not match some triangles
                // break;
            }
        }
        num_visited_cells[i] = jc;
    }
}

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
                             const float eps) {
    const size_t block_size = 32;
    post_process_tetrahedra_kernel<<<(num_rays + block_size - 1) / block_size, block_size>>>(num_rays,
                                                                                             max_visited_triangles,
                                                                                             max_visited_cells,
                                                                                             num_visited_triangles,
                                                                                             visited_triangles,
                                                                                             barycentric_coordinates_triangle,
                                                                                             distances,
                                                                                             visited_cells,
                                                                                             num_visited_cells,
                                                                                             barycentric_coordinates,
                                                                                             cell_distances,
                                                                                             eps);
}

__global__ void find_matched_cells_kernel(const size_t num_rays,
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
                                          float4 *barycentric_coordinates_out) {
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
                if (matched_vertices_out != nullptr) {
                    // NOTE, there is nonlocal access
                    // This could make the kernel slower, therfore reporting vertices is optional
                    matched_vertices_out[i * num_samples_per_ray + j] = cells[visited_cell];
                }

                float mult = (current_distance - current_hit_distance.x) / (current_hit_distance.y - current_hit_distance.x);
                const float4 coords1 = barycentric_coordinates[(i * max_visited_cells + current_cell_pointer) * 2];
                const float4 coords2 = barycentric_coordinates[(i * max_visited_cells + current_cell_pointer) * 2 + 1];
                barycentric_coordinates_out[i * num_samples_per_ray + j] = make_float4(
                    mult * coords1.x + (1 - mult) * coords2.x,
                    mult * coords1.y + (1 - mult) * coords2.y,
                    mult * coords1.z + (1 - mult) * coords2.z,
                    mult * coords1.w + (1 - mult) * coords2.w);
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
                        const float4 *barycentric_coordinates,
                        const float *distances,
                        unsigned int *matched_cells_out,
                        uint4 *matched_vertices_out,
                        bool *mask_out,
                        float4 *barycentric_coordinates_out) {
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
        matched_cells_out,
        matched_vertices_out,
        mask_out,
        barycentric_coordinates_out);
}

__global__ void interpolate_values_kernel(const size_t num_values,
                                          const size_t field_dim,
                                          const uint4 *vertex_indices,
                                          const float4 *barycentric_coordinates,
                                          const float *field,
                                          float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < num_values; i += stride) {
        for (size_t j = 0; j < field_dim; ++j) {
            result[i * field_dim + j] = (barycentric_coordinates[i].x * field[vertex_indices[i].x * field_dim + j] +
                                         barycentric_coordinates[i].y * field[vertex_indices[i].y * field_dim + j] +
                                         barycentric_coordinates[i].z * field[vertex_indices[i].z * field_dim + j] +
                                         barycentric_coordinates[i].w * field[vertex_indices[i].w * field_dim + j]);
        }
    }
}

__global__ void interpolate_values_backward_kernel(const size_t num_vertices,
                                                   const size_t num_values,
                                                   const size_t field_dim,
                                                   const uint4 *vertex_indices,
                                                   const float4 *barycentric_coordinates,
                                                   const float *grad_in,
                                                   float *field_grad_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < num_values; i += stride) {
        for (size_t j = 0; j < field_dim; ++j) {
            atomicAdd(&field_grad_out[vertex_indices[i].x * field_dim + j], barycentric_coordinates[i].x * grad_in[i * field_dim + j]);
            atomicAdd(&field_grad_out[vertex_indices[i].y * field_dim + j], barycentric_coordinates[i].y * grad_in[i * field_dim + j]);
            atomicAdd(&field_grad_out[vertex_indices[i].z * field_dim + j], barycentric_coordinates[i].z * grad_in[i * field_dim + j]);
            atomicAdd(&field_grad_out[vertex_indices[i].w * field_dim + j], barycentric_coordinates[i].w * grad_in[i * field_dim + j]);
        }
    }
}

void interpolate_values(const size_t num_values,
                        const size_t field_dim,
                        const uint4 *vertex_indices,
                        const float4 *barycentric_coordinates,
                        const float *field,
                        float *result) {
    const size_t block_size = 1024;
    interpolate_values_kernel<<<(num_values + block_size - 1) / block_size, block_size>>>(
        num_values,
        field_dim,
        vertex_indices,
        barycentric_coordinates,
        field,
        result);
}

void interpolate_values_backward(const size_t num_vertices,
                                 const size_t num_values,
                                 const size_t field_dim,
                                 const uint4 *vertex_indices,
                                 const float4 *barycentric_coordinates,
                                 const float *grad_in,
                                 float *field_grad_out) {
    const size_t block_size = 1024;
    interpolate_values_backward_kernel<<<(num_values + block_size - 1) / block_size, block_size>>>(
        num_vertices,
        num_values,
        field_dim,
        vertex_indices,
        barycentric_coordinates,
        grad_in,
        field_grad_out);
}
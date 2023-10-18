#include <optix.h>

#include "../optix_types.h"
#include "../utils/vec_math.h"

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

struct float6 {
    float x, y, z, w, u, v;
};

extern "C" __forceinline__ __device__ bool get_common_tetrahedra(const uint2 &tetrahedra1, const uint2 &tetrahedra2, unsigned int &tetrahedra) {
    if (tetrahedra1.x == tetrahedra2.x) {
        tetrahedra = tetrahedra1.x;
        return true;
    } else if (tetrahedra1.x == tetrahedra2.y) {
        tetrahedra = tetrahedra1.x;
        return true;
    } else if (tetrahedra1.y == tetrahedra2.x) {
        tetrahedra = tetrahedra1.y;
        return true;
    } else if (tetrahedra1.y == tetrahedra2.y) {
        tetrahedra = tetrahedra1.y;
        return true;
    }
    return false;
}

extern "C" __forceinline__ __device__ uint4 combine_indices(
    const uint3 &id1,
    const uint3 &id2,
    const float2 &coords_in1,
    const float2 &coords_in2,
    float3 &coords_out1,
    float3 &coords_out2) {
    uint4 result = make_uint4(0, id1.x, id1.y, id1.z);
    coords_out1 = make_float3(1.0f - coords_in1.x - coords_in1.y, coords_in1.x, coords_in1.y);
    float3 out2_ref = make_float3(1.0f - coords_in2.x - coords_in2.y, coords_in2.x, coords_in2.y);
    coords_out2 = make_float3(0, 0, 0);


    unsigned char matched = 0;

#pragma unroll
    for (unsigned char i = 0; i < 3; ++i) {
        unsigned int idx2 = ((unsigned int *)&id2)[i];
        float mult2 = ((float *)&out2_ref)[i];
        bool was_break = false;
#pragma unroll
        for (unsigned char j = 0; j < 3; ++j) {
            unsigned int idx1 = ((unsigned int *)&id1)[j];
            if (idx1 == idx2) {
                ((float *)&coords_out2)[j] = mult2;
                matched++;
                was_break = true;
                break;
            }
        }
        if (!was_break) {
            // This is the new index
            result.x = idx2;
        }
    }
    return result;
}


template <typename V1, typename V2>
__forceinline__ __device__ void bitonic_sort(unsigned int N,
                                             float2 *distances,
                                             V1 *values1,
                                             V2 *values2) {
    int Nup2 = 1;
    while (Nup2 < N)
        Nup2 = Nup2 << 1;

    for (int i = N; i < Nup2; i++)
        distances[i].x = 1e20f;
    N = Nup2;

    for (int k = 2; k <= N; k = k << 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            for (int i = 0; i < N; i++) {
                const int ij = i ^ j;
                if (ij > i) {
                    const int ik = i & k;
                    const float tmp_d = distances[i].x;
                    if ((ik == 0 && tmp_d > distances[ij].x) ||  // sort ascending
                        (ik != 0 && tmp_d < distances[ij].x)) {  // sort descending
                        swap(distances[i], distances[ij]);
                        swap(values1[i], values1[ij]);
                        swap(values2[i], values2[ij]);
                    }
                }
            }
        }
    }
}

extern "C" __forceinline__ __device__ void post_process_tetrahedra(unsigned int i) {
    // Dedupe triangles
    // TODO: do this more efficiently!!
    const unsigned int ray_len = params.num_visited_tetrahedra[i];
    size_t jc = 0;  // Local index to next cell to write


    const unsigned int max_visited_triangles = params.max_ray_triangles;
    constexpr unsigned int empty = ~((unsigned int)0u);
    auto t = params.visited_tetrahedra + (i * max_visited_triangles);
    const auto dl = params.hit_distances + (i * max_visited_triangles);
    const auto bcs = params.barycentric_coordinates + ((i * max_visited_triangles) * 2);
    // We remove empty tetrahedra
    // Using hit_distances.y for bookkeeping
    for (size_t j = 0; j + 1 < ray_len; ++j) {
        if (t[j] == empty) {
            continue;
        }
        float dn = dl[j].x;
        bool clear_self = false;
        for (size_t offset = 1; j + offset < ray_len && (t[j + offset] == empty || abs(dl[j + offset].x - dn) < eps); offset++) {
            const auto tj_cell = params.triangle_tetrahedra[t[j]];
            const auto tj_offset_cell = params.triangle_tetrahedra[t[j + offset]];
            unsigned int cell;
            if (t[j + offset] != empty && get_common_tetrahedra(tj_cell, tj_offset_cell, cell)) {
                // Triangle registered two times
                // We only count it once, there can be exitting face later on
                if (t[j] != t[j + offset]) {
                    // This is the case where the tetrahedra was hitted twice
                    // But the length of the intersection is too small to register
                    // We ignore the cell altogether
                    clear_self = true;
                }
                if (dl[j + offset].y > 0.0f) {
                    // Already marked for deletion once
                    t[j + offset] = empty;
                } else {
                    // Mark for deletion
                    dl[j + offset].y = 1.0f;
                }
            }
        }
        if (clear_self) {
            if (dl[j].y > 0.0f) {
                // Already marked for deletion once
                t[j] = empty;
            }
        }
        dl[j].y = 0.0f;
    }

    // printf("triangles: ");
    // for (int j = 0; j < ray_len; ++j) {
    //     const uint tt = params.visited_tetrahedra[i * params.max_ray_triangles + j];
    //     printf("%d (%.2f) [%u,%u], ", params.visited_tetrahedra[i * params.max_ray_triangles + j], params.hit_distances[i * params.max_ray_triangles + j].x,
    //     params.triangle_tetrahedra[tt].x, params.triangle_tetrahedra[tt].y);
    // }
    // printf("\n");


    // Test triangles only
    // for (size_t j = 0; j < ray_len; ++j) {
    //     const auto jc_g = i * max_visited_triangles + jc;
    //     if (t[j] == empty) {
    //         continue;
    //     }
    //     params.hit_distances[jc_g] = make_float2(dl[j].x, dl[j].x);
    //     const auto bc = bcs[2*j];
    //     params.barycentric_coordinates[2*jc_g] = make_float3(1.0f - bc.x - bc.y, bc.x, bc.y);
    //     params.barycentric_coordinates[2*jc_g+1] = params.barycentric_coordinates[2*jc_g];
    //     params.visited_tetrahedra[jc_g] = t[j];
    //     const auto ti = params.triangle_indices[t[j]];
    //     params.vertex_indices[jc_g] = make_uint4(0, ti.x, ti.y, ti.z);
    //     jc++;
    // }
    // params.num_visited_tetrahedra[i] = jc;
    // return;

    for (size_t j = 0; j < ray_len; ++j) {
        if (t[j] == empty) {
            continue;
        }

        const auto orig_tj = params.triangle_tetrahedra[t[j]];
        auto dn = dl[j].x;
        size_t real_offset = 1;
        // bool was_error = true;
        for (size_t offset = 1;
             j + offset < ray_len && (real_offset < 3 || t[j + offset] == empty || abs(dl[j + offset].x - dn) < eps);
             offset++) {
            if (t[j + offset] == empty) {
                continue;
            }

            unsigned int cell;
            if (get_common_tetrahedra(orig_tj, params.triangle_tetrahedra[t[j + offset]], cell)) {
                // Close face - dump to output
                // Output only if tetrahedron is nonempty
                if (abs(dl[j].x - dl[j + offset].x) >= eps) {
                    const auto jc_g = i * max_visited_triangles + jc;
                    const float2 coords0 = *(float2*)(bcs + 2*j);
                    const float2 coords1 = *(float2*)(bcs + 2*(j + offset));
                    float3 &bc0 = *(params.barycentric_coordinates + 2*jc_g);
                    float3 &bc1 = *(params.barycentric_coordinates + 2*jc_g + 1);
                    // printf("%.2f %.2f (%d) %.2f %.2f\n", coords0.x, coords0.y, offset, coords1.x, coords1.y);

                    uint4 vertex_indices = combine_indices(
                        params.triangle_indices[t[j]], params.triangle_indices[t[j + offset]],
                        coords0, coords1,
                        bc0, bc1);
                    if (params.vertex_indices != nullptr)
                        params.vertex_indices[jc_g] = vertex_indices;
                    
                    params.hit_distances[jc_g] = make_float2(dl[j].x, dl[j + offset].x);
                    params.visited_tetrahedra[jc_g] = cell;
                    jc++;
                }

                // Swap next
                if (offset > 1) {
                    swap(dl[j + offset], dl[j + 1]);
                    swap(bcs[(j + offset)*2], bcs[(j + 1)*2]);
                    // No need to swap this - it will be empty
                    // swap(bcs[(j + offset)*2 + 1], bcs[(j + 1)*2 + 1]);
                    swap(t[j + offset], t[j + 1]);
                }

                // was_error = false;
                break;
            }
            dn = dl[j + offset].x;
            real_offset++;
        }
        // if (was_error && real_offset > 1) {
        //     // If real offest == 1, this is the last triangle on the ray
        //     // The ray might have been terminated prematurely because
        //     // the max_visited_triangles wasn't large enough
        //     // Therefore, this should not be considered an error.
        //     // if (i == 97810) {
        //     // printf("error on ray: %d\n", i);
        //     // for (size_t j = 0; j < jc; ++j) {
        //     //     printf("%05d \n", visited_cells[i*max_visited_cells+j]);
        //     // }
        //     // }
        //     // There was an error, we could not match some triangles
        //     // break;
        // }
    }

    // Empty our the rest of the ray
    for (size_t j = jc; j < max_visited_triangles; ++j) {
        params.visited_tetrahedra[i * max_visited_triangles + j] = empty;
        if (params.vertex_indices != nullptr)
            params.vertex_indices[i * max_visited_triangles + j] = make_uint4(empty, empty, empty, empty);
    }
    params.num_visited_tetrahedra[i] = jc;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[idx.x];
    const float3 ray_direction = params.ray_directions[idx.x];

    // Trace the ray against our scene hierarchy
    unsigned int p0 = 0;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // rayTime -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset   -- See SBT discussion
        1,  // SBT stride   -- See SBT discussion
        0,  // missSBTIndex -- See SBT discussion
        p0);

    params.num_visited_tetrahedra[idx.x] = p0;

    bitonic_sort(p0,
                 params.hit_distances + idx.x * params.max_ray_triangles,
                 params.visited_tetrahedra + idx.x * params.max_ray_triangles,
                 ((float6*)params.barycentric_coordinates) + idx.x * params.max_ray_triangles);

    post_process_tetrahedra(idx.x);
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ms() {
}

extern "C" __global__ void __anyhit__ms() {
    const unsigned int num_triangles = optixGetPayload_0();
    if (num_triangles >= params.max_ray_triangles - 1) {
        optixTerminateRay();
        return;
    }

    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    // const float2 barycentrics = optixGetTriangleBarycentrics();
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int current_triangle = optixGetPrimitiveIndex();
    params.visited_tetrahedra[idx.x * params.max_ray_triangles + num_triangles] = current_triangle;
    params.hit_distances[idx.x * params.max_ray_triangles + num_triangles] = make_float2(optixGetRayTmax(), 0);
    const auto coordinates = optixGetTriangleBarycentrics();
    params.barycentric_coordinates[2 * (idx.x * params.max_ray_triangles + num_triangles)] = make_float3(coordinates.x, coordinates.y, 0);

    // setPayload(make_float3(barycentrics, 1.0f));
    optixSetPayload_0(num_triangles + 1);
    optixIgnoreIntersection();
}
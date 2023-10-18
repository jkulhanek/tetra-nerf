// #include <optix.h>
//
// #include "optix_types.h"
// #include "utils/vec_math.h"
//
// extern "C" __global__ void __raygen__findtetrahedron() {
//     // Lookup our location within the launch grid
//     const uint3 idx = optixGetLaunchIndex();
//     const uint3 dim = optixGetLaunchDimensions();
//
//     // Map our launch idx to a screen location and create a ray from the camera
//     // location through the screen
//     const float3 ray_origin = params.ray_origins[idx.x];
//     const float3 ray_direction = params.ray_directions[idx.x];
//
// }
//
// extern "C" __global__ void __closesthit__findtetrahedron() {
// }
#include <limits.h>
#include <optix.h>

#include "../optix_types.h"
#include "../utils/vec_math.h"

extern "C" {
__constant__ ParamsFindTetrahedra params;
constexpr unsigned int uint_max = ~((unsigned int)0u);
}

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

extern "C" __global__ void __raygen__ft() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[idx.x];

    // unsigned int max_pow2 = 2;
    // while (max_pow2 < params.max_visited_triangles) {
    //     max_pow2 <<= 1;
    // }
    // max_pow2 >>= 1;

    // Trace the ray against our scene hierarchy
    // Trace the ray against our scene hierarchy
    unsigned int p0 = uint_max;
    unsigned int _a, _b, _dist;
    float a0 = 0;
    float b0 = 0;
    float dist0 = 0;
    optixTrace(
        params.handle,
        ray_origin,
        make_float3(1.0f, 0.0f, 0.0f),
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // rayTime -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset   -- See SBT discussion
        1,  // SBT stride   -- See SBT discussion
        0,  // missSBTIndex -- See SBT discussion
        p0, _a, _b, _dist);
    a0 = __uint_as_float(_a);
    b0 = __uint_as_float(_b);
    dist0 = __uint_as_float(_dist);

    unsigned int p1 = uint_max;
    float a1 = 0;
    float b1 = 0;
    float dist1 = 0;
    optixTrace(
        params.handle,
        ray_origin,
        make_float3(-1.0f, 0.0f, 0.0f),
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // rayTime -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset   -- See SBT discussion
        1,  // SBT stride   -- See SBT discussion
        0,  // missSBTIndex -- See SBT discussion
        p1, _a, _b, _dist);
    a1 = __uint_as_float(_a);
    b1 = __uint_as_float(_b);
    dist1 = __uint_as_float(_dist);
    
    // printf("p0: %u (%.2f), p1: %u (%.2f) \n", p0, dist0, p1, dist1);

    // printf("p0: %u (%u, %u, %u) (%u, %u), p1: %u (%u, %u, %u) (%u, %u) \n",
    //        p0,
    //        params.triangle_indices[p0].x,
    //        params.triangle_indices[p0].y,
    //        params.triangle_indices[p0].z,
    //        params.triangle_tetrahedra[p0].x,
    //        params.triangle_tetrahedra[p0].y,
    //        p1,
    //        params.triangle_indices[p1].x,
    //        params.triangle_indices[p1].y,
    //        params.triangle_indices[p1].z,
    //        params.triangle_tetrahedra[p1].x,
    //        params.triangle_tetrahedra[p1].y);
    // printf("cin0: (%u, %u, %u) (%.2f, %.2f, %.2f) (%.2f)\n",
    //        params.triangle_indices[p0].x,
    //        params.triangle_indices[p0].y,
    //        params.triangle_indices[p0].z,
    //        1 - a0 - b0, a0, b0, dist0);
    // printf("cin1: (%u, %u, %u) (%.2f, %.2f, %.2f) (%.2f)\n",
    //        params.triangle_indices[p1].x,
    //        params.triangle_indices[p1].y,
    //        params.triangle_indices[p1].z,
    //        1 - a1 - b1, a1, b1, dist1);

    float3 coords0;
    float3 coords1;
    unsigned int cell = uint_max;
    if (p0 != uint_max && p1 != uint_max && get_common_tetrahedra(params.triangle_tetrahedra[p0], params.triangle_tetrahedra[p1], cell)) {
        uint4 indices = combine_indices(
            params.triangle_indices[p0],
            params.triangle_indices[p1],
            make_float2(a0, b0),
            make_float2(a1, b1),
            coords0,
            coords1);
        float m = dist1 / (dist0 + dist1);

        float3 coords = make_float3(
            coords0.x * m + coords1.x * (1 - m),
            coords0.y * m + coords1.y * (1 - m),
            coords0.z * m + coords1.z * (1 - m));
        params.barycentric_coordinates[idx.x] = coords;
        params.vertex_indices[idx.x] = indices;
        // printf("found tetrahedra: %d (%u, %u, %u, %u), (%.2f, %.2f, %.2f, %.2f)\n",
        //        cell,
        //        indices.x,
        //        indices.y,
        //        indices.z,
        //        indices.w,
        //        1 - coords.x - coords.y - coords.z,
        //        coords.x,
        //        coords.y,
        //        coords.z);
    }
    params.tetrahedra[idx.x] = cell;
}

extern "C" __global__ void __miss__ft() {
}

extern "C" __global__ void __closesthit__ft() {
    const unsigned int current_triangle = optixGetPrimitiveIndex();
    optixSetPayload_0(current_triangle);
    const auto barycentric_coords = optixGetTriangleBarycentrics();
    optixSetPayload_1(__float_as_uint(barycentric_coords.x));
    optixSetPayload_2(__float_as_uint(barycentric_coords.y));
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
}
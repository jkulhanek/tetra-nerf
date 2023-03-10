#include <optix.h>

#include "optix_types.h"
#include "utils/vec_math.h"

extern "C" {
__constant__ Params params;
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename Key, typename V1, typename V2>
__forceinline__ __device__ void quicksort(const unsigned int n,
                                          Key *distances,
                                          V1 *values1,
                                          V2 *values2) {
    // Sorts a (portion of an) array, divides it into partitions, then sorts those
    if (n < 2) {
        return;
    }
    int lo = 0;
    int hi = n - 1;
    Key *stack = new Key[(hi - lo + 1) * 2];

    // initialize top of stack
    int top = 0;

    stack[top++] = lo;
    stack[top++] = hi;
    while (top > 0) {
        // Pop hi and lo
        hi = stack[--top];
        lo = stack[--top];
        int p = (hi + lo) / 2;
        Key pivot = distances[p];  // The value in the middle of the array

        // Left index
        int i = lo;

        // Right index
        int j = hi;

        while (i <= j) {
            while (distances[i] < pivot) i++;
            while (distances[j] > pivot) j--;

            if (i <= j) {
                // Swap the elements at the left and right indices
                swap(distances[i], distances[j]);
                swap(values1[i], values1[j]);
                swap(values2[i], values2[j]);
                i++;
                j--;
            }
        }

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (lo < j) {
            stack[top++] = lo;
            stack[top++] = j;
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (i < hi) {
            stack[top++] = i;
            stack[top++] = hi;
        }
    }
    delete stack;
}

template <typename Key, typename V1, typename V2>
__forceinline__ __device__ void bitonic_sort(unsigned int N,
                                             Key *distances,
                                             V1 *values1,
                                             V2 *values2) {
    int Nup2 = 1;
    while (Nup2 < N)
        Nup2 = Nup2 << 1;

    for(int i=N; i<Nup2; i++)
        distances[i] = 1e20f;
    N = Nup2;

    for (int k = 2; k <= N; k = k << 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            for (int i = 0; i < N; i++) {
                const int ij = i ^ j;
                if (ij > i) {
                    const int ik = i & k;
                    const Key tmp_d = distances[i];
                    if ((ik == 0 && tmp_d > distances[ij]) ||  // sort ascending
                        (ik != 0 && tmp_d < distances[ij])) {  // sort descending
                        swap(distances[i], distances[ij]);
                        swap(values1[i], values1[ij]);
                        swap(values2[i], values2[ij]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[idx.x];
    const float3 ray_direction = params.ray_directions[idx.x];

    // unsigned int max_pow2 = 2;
    // while (max_pow2 < params.max_visited_triangles) {
    //     max_pow2 <<= 1;
    // }
    // max_pow2 >>= 1;

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

    // Record results in our output raster
    // params.image[idx.x] = make_color(result);
    params.shared_visited_triangles[idx.x] = p0;

    // if (idx.x == 97810) {
    // if(idx.x == 67013) {
    //     for (size_t i = 0; i < p0; ++i) {
    //         printf("%d %.4f \n", params.visited_triangles[idx.x * params.max_visited_triangles + i] / 4, params.hit_distances[idx.x * params.max_visited_triangles + i]);
    //     }
    // }

    // if (p0 <= max_pow2) {
    unsigned int * visited_triangles = params.shared_visited_triangles + dim.x;
    bitonic_sort(p0,
                params.hit_distances + idx.x * params.max_ray_triangles,
                visited_triangles + idx.x * params.max_ray_triangles,
                params.barycentric_coordinates + idx.x * params.max_ray_triangles);
    // } else {
    //     quicksort(p0,
    //               params.hit_distances + idx.x * params.max_visited_triangles,
    //               params.visited_triangles + idx.x * params.max_visited_triangles,
    //               params.barycentric_coordinates + idx.x * params.max_visited_triangles);
    //}

    // if (idx.x == 97810) {
    // if(idx.x == 67013) {
    //     for (size_t i = 0; i < p0; ++i) {
    //         printf("(%.12f, %d, %d),\n", params.hit_distances[idx.x * params.max_visited_triangles + i], params.visited_triangles[idx.x * params.max_visited_triangles + i] / 4, params.visited_triangles[idx.x * params.max_visited_triangles + i] %4);
    //     }
    // }
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ms() {
}

extern "C" __global__ void __anyhit__ms() {
    const unsigned int num_triangles = optixGetPayload_0();
    if (num_triangles >= params.max_visited_cells*2 - 1) {
        optixTerminateRay();
        return;
    }

    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    // const float2 barycentrics = optixGetTriangleBarycentrics();
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int current_triangle = optixGetPrimitiveIndex();
    params.shared_visited_triangles[dim.x + idx.x * params.max_ray_triangles + num_triangles] = current_triangle;
    params.hit_distances[idx.x * params.max_ray_triangles + num_triangles] = optixGetRayTmax();
    params.barycentric_coordinates[idx.x * params.max_ray_triangles + num_triangles] = optixGetTriangleBarycentrics();

    // setPayload(make_float3(barycentrics, 1.0f));
    optixSetPayload_0(num_triangles + 1);
    optixIgnoreIntersection();
}
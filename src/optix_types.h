struct Params {
    const float3* ray_origins;
    const float3* ray_directions;
    const uint3 *triangle_indices;
    const uint2 *triangle_tetrahedra;

    unsigned int* num_visited_tetrahedra;
    unsigned int* visited_tetrahedra;
    uint4 *vertex_indices;
    float3* barycentric_coordinates;
    float2* hit_distances;
    unsigned int max_ray_triangles;
    OptixTraversableHandle handle;
};

struct RayGenData {
    // No data needed
};

struct MissData {
    // No data needed
};

struct HitGroupData {
    // No data needed
};

struct ParamsFindTetrahedra {
    const uint3* triangle_indices;
    const uint2* triangle_tetrahedra;
    const float3* ray_origins;
    uint4* vertex_indices;
    float3* barycentric_coordinates;
    unsigned int* tetrahedra;
    OptixTraversableHandle handle;
};

struct ParamsTraceRaysTriangles {
    const float3* ray_origins;
    const float3* ray_directions;
    const uint3 *triangle_indices;
    const uint2 *triangle_tetrahedra;

    unsigned int* num_visited_triangles;
    unsigned int* visited_triangles;
    uint3 *vertex_indices;
    float2* barycentric_coordinates;
    float* hit_distances;
    unsigned int max_ray_triangles;
    OptixTraversableHandle handle;
};
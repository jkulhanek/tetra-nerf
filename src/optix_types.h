struct Params {
    const float3* ray_origins;
    const float3* ray_directions;
    unsigned int* shared_visited_triangles;
    float2* barycentric_coordinates;
    float* hit_distances;
    unsigned int max_ray_triangles;
    unsigned int max_visited_cells;
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

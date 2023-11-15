#include "tetrahedra_tracer.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "optix_types.h"
#include "utils/exception.h"
#include "utils/tensor.h"
#include "utils/vec_math.h"

namespace fs = std::filesystem;

uint3 order_faces(const uint3 &face) {
    uint3 ordered_face = face;
    if (ordered_face.x > ordered_face.y) {
        std::swap(ordered_face.x, ordered_face.y);
    }
    if (ordered_face.y > ordered_face.z) {
        std::swap(ordered_face.y, ordered_face.z);
    }
    if (ordered_face.x > ordered_face.y) {
        std::swap(ordered_face.x, ordered_face.y);
    }
    return ordered_face;
}

template <>
struct std::hash<uint3> {
    std::size_t operator()(const uint3 &k) const {
        using std::hash;
        using std::size_t;
        using std::string;
        return ((hash<unsigned int>()(k.x) ^ (hash<unsigned int>()(k.y) << 1)) >> 1) ^ (hash<unsigned int>()(k.z) << 1);
    }
};

void convert_tetrahedra_to_triangles(const size_t num_tetrahedra,
                                     const uint4 *tetrahedra,
                                     std::vector<uint3> &triangles,
                                     std::vector<uint2> &triangles_tetrahedra) {
    unsigned int empty = ~((unsigned int)0);
    std::unordered_map<uint3, unsigned int> known_faces;
    for (size_t i = 0; i < num_tetrahedra; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint4 tetrahedron = tetrahedra[i];
            uint3 triangle = make_uint3(
                ((unsigned int *)&tetrahedron)[(j + 1) % 4],
                ((unsigned int *)&tetrahedron)[(j + 2) % 4],
                ((unsigned int *)&tetrahedron)[(j + 3) % 4]);
            auto ordered_triangle = order_faces(triangle);
            if (known_faces.find(ordered_triangle) == known_faces.end()) {
                known_faces[ordered_triangle] = triangles_tetrahedra.size();
                triangles.push_back(triangle);
                triangles_tetrahedra.push_back(make_uint2(i, empty));
            } else {
                if (triangles_tetrahedra[known_faces[ordered_triangle]].y != empty) {
                    throw std::runtime_error("A triangle is shared by more than two tetrahedra!");
                }
                triangles_tetrahedra[known_faces[ordered_triangle]].y = i;
            }
        }
    }
}

// These structs represent the data blocks of our SBT records
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// TODO: move to python
static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

TetrahedraTracer::TetrahedraTracer(int8_t device) : device(device) {
    // Initialize fields
    context = nullptr;

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        // Warning: CUDA should have been already initialized at this point!!
        CUDA_CHECK(cudaFree(0));

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    trace_rays_pipeline = std::move(TraceRaysPipeline(context, device));
    trace_rays_triangles_pipeline = std::move(TraceRaysTrianglesPipeline(context, device));
    find_tetrahedra_pipeline = std::move(FindTetrahedraPipeline(context, device));
    tetrahedra_structure = std::move(TetrahedraStructure(context, device));
}

TetrahedraTracer::TetrahedraTracer(TetrahedraTracer &&other)
    : context(std::exchange(other.context, nullptr)),
        device(std::exchange(other.device, -1)),
        tetrahedra_structure(std::move(other.tetrahedra_structure)),
        find_tetrahedra_pipeline(std::move(other.find_tetrahedra_pipeline)),
        trace_rays_pipeline(std::move(other.trace_rays_pipeline)),
        trace_rays_triangles_pipeline(std::move(other.trace_rays_triangles_pipeline)) {}

void TraceRaysPipeline::trace_rays(const TetrahedraStructure *tetrahedra_structure,
                                   const size_t num_rays,
                                   const unsigned int max_ray_triangles,
                                   const float3 *ray_origins,
                                   const float3 *ray_directions,
                                   unsigned int *num_visited_cells_out,
                                   unsigned int *visited_cells_out,
                                   float3 *barycentric_coordinates_out,
                                   float2 *hit_distances_out,
                                   uint4 *vertex_indices_out) {
    CUDA_CHECK(cudaSetDevice(device));
    float2 *triangle_barycentric_coordinates;
    unsigned int *shared_visited_triangles;
    float *triangle_hit_distances;

    // TODO: we can reuse the triangle memory
    // No need to allocate again here
    {
        Params params;
        params.triangle_tetrahedra = tetrahedra_structure->triangle_tetrahedra();
        params.triangle_indices = tetrahedra_structure->triangle_indices();
        params.num_visited_tetrahedra = num_visited_cells_out;
        params.visited_tetrahedra = visited_cells_out;
        params.vertex_indices = vertex_indices_out;
        params.max_ray_triangles = max_ray_triangles;
        params.barycentric_coordinates = barycentric_coordinates_out;
        params.hit_distances = hit_distances_out;
        params.handle = tetrahedra_structure->gas_handle();
        params.ray_origins = ray_origins;
        params.ray_directions = ray_directions;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt, num_rays, 1, 1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

TetrahedraTracer::~TetrahedraTracer() noexcept(false) {
    // We call the destructor manually here to ensure correct destruction order
    tetrahedra_structure.~TetrahedraStructure();
    trace_rays_pipeline.~TraceRaysPipeline();
    trace_rays_triangles_pipeline.~TraceRaysTrianglesPipeline();
    find_tetrahedra_pipeline.~FindTetrahedraPipeline();

    if (context != nullptr && device != -1) {
        CUDA_CHECK(cudaSetDevice(device));
        OPTIX_CHECK(optixDeviceContextDestroy(std::exchange(context, nullptr)));
    }
}

TetrahedraStructure::TetrahedraStructure() noexcept
    : device(-1),
      context(nullptr),
      gas_handle_(0),
      d_gas_output_buffer(0),
      num_vertices(0),
      num_cells(0),
      tetrahedra_vertices(nullptr),
      triangle_indices_(nullptr),
      triangle_tetrahedra_(nullptr) {}


TetrahedraStructure::TetrahedraStructure(TetrahedraStructure &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      gas_handle_(std::exchange(other.gas_handle_, 0)),
      d_gas_output_buffer(std::exchange(other.d_gas_output_buffer, 0)),
      num_vertices(std::exchange(other.num_vertices, 0)),
      num_cells(std::exchange(other.num_cells, 0)),
      tetrahedra_vertices(std::exchange(other.tetrahedra_vertices, nullptr)),
      triangle_indices_(std::exchange(other.triangle_indices_, nullptr)),
      triangle_tetrahedra_(std::exchange(other.triangle_tetrahedra_, nullptr)) {}

void TetrahedraStructure::release() {
    bool device_set = false;
    if (d_gas_output_buffer != 0) {
        if (!device_set) { CUDA_CHECK(cudaSetDevice(device)); device_set = true; }
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));
        d_gas_output_buffer = 0;
    }
    gas_handle_ = 0;

    // NOTE: we do not own the vertices and cells arrays
    tetrahedra_vertices = nullptr;
    if (triangle_indices_ != nullptr) {
        if (!device_set) { CUDA_CHECK(cudaSetDevice(device)); device_set = true; }
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(triangle_indices_)));
        triangle_indices_ = nullptr;
    }
    if (triangle_tetrahedra_ != nullptr) {
        if (!device_set) { CUDA_CHECK(cudaSetDevice(device)); device_set = true; }
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(triangle_tetrahedra_)));
        triangle_tetrahedra_ = nullptr;
    }
}

TetrahedraStructure::~TetrahedraStructure() noexcept(false) {
    if (this->device != -1) {
        release();
    }
    const auto device = std::exchange(this->device, -1);
}

void TetrahedraStructure::build(const size_t num_vertices,
                                const size_t num_cells,
                                const float3 *d_vertices,
                                const uint4 *cells) {
    release();

    CUDA_CHECK(cudaSetDevice(device));

    // Build list of triangles from tetrahedra
    unsigned int num_triangles;
    {
        uint4 *h_cells = new uint4[num_cells];
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(h_cells),
            cells, num_cells * sizeof(uint4),
            cudaMemcpyDeviceToHost));
        std::vector<uint3> h_triangle_indices;
        std::vector<uint2> h_triangle_tetrahedra;
        convert_tetrahedra_to_triangles(num_cells, h_cells, h_triangle_indices, h_triangle_tetrahedra);
        delete[] h_cells;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&triangle_tetrahedra_),
            h_triangle_tetrahedra.size() * sizeof(uint2)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(triangle_tetrahedra_),
            h_triangle_tetrahedra.data(),
            h_triangle_tetrahedra.size() * sizeof(uint2),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&triangle_indices_),
            h_triangle_indices.size() * sizeof(uint3)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(triangle_indices_),
            h_triangle_indices.data(),
            h_triangle_indices.size() * sizeof(uint3),
            cudaMemcpyHostToDevice));
        num_triangles = h_triangle_indices.size();
    }

    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr *>(&d_vertices);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(num_vertices);

    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(triangle_indices_);
    triangle_input.triangleArray.numIndexTriplets = num_triangles;

    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &triangle_input,
        1,  // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        nullptr,  // emitted property list
        0         // num emitted properties
        ));
    this->num_vertices = num_vertices;
    this->num_cells = num_cells;
    this->tetrahedra_vertices = d_vertices;

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
}

TraceRaysPipeline::TraceRaysPipeline(const OptixDeviceContext &context, int8_t device) : device(device), context(context) {
    // Initialize fields
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Create module
    //
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

// The following is not supported in Optix 7.2
// #define XSTR(x) STR(x)
// #define STR(x) #x
// #pragma message "Optix ABI version is: " XSTR(OPTIX_ABI_VERSION)
#if (OPTIX_ABI_VERSION > 54)
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 1;
        // https://forums.developer.nvidia.com/t/how-to-calculate-numattributevalues-of-optixpipelinecompileoptions/110833
        pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        std::string input = TraceRaysPipeline::load_ptx_data();
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input.c_str(),
            input.size(),
            log,
            &sizeof_log,
            &module));
    }

    //
    // Create program groups
    //
    {
        OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};  //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ms";
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0,  // maxCCDepth
                                               0,  // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1  // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size,
            cudaMemcpyHostToDevice));

        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;
    }

    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    }
}


TraceRaysPipeline::TraceRaysPipeline(TraceRaysPipeline &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      pipeline(std::exchange(other.pipeline, nullptr)),
      raygen_prog_group(std::exchange(other.raygen_prog_group, nullptr)),
      miss_prog_group(std::exchange(other.miss_prog_group, nullptr)),
      hitgroup_prog_group(std::exchange(other.hitgroup_prog_group, nullptr)),
      module(std::exchange(other.module, nullptr)),
      sbt(std::exchange(other.sbt, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}
      


TraceRaysPipeline::~TraceRaysPipeline() noexcept(false) {
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device));
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (sbt.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
    if (sbt.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
    sbt = {};
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}

FindTetrahedraPipeline::FindTetrahedraPipeline(const OptixDeviceContext &context, int8_t device) : device(device), context(context) {
    // Initialize fields
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Create module
    //
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

// The following is not supported in Optix 7.2
// #define XSTR(x) STR(x)
// #define STR(x) #x
// #pragma message "Optix ABI version is: " XSTR(OPTIX_ABI_VERSION)
#if (OPTIX_ABI_VERSION > 54)
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 4;
        // https://forums.developer.nvidia.com/t/how-to-calculate-numattributevalues-of-optixpipelinecompileoptions/110833
        pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        std::string input = load_ptx_data();
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input.c_str(),
            input.size(),
            log,
            &sizeof_log,
            &module));
    }

    //
    // Create program groups
    //
    {
        OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};  //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__ft";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ft";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ft";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0,  // maxCCDepth
                                               0,  // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1  // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size,
            cudaMemcpyHostToDevice));

        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(ParamsFindTetrahedra)));
}

FindTetrahedraPipeline::FindTetrahedraPipeline(FindTetrahedraPipeline &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      pipeline(std::exchange(other.pipeline, nullptr)),
      raygen_prog_group(std::exchange(other.raygen_prog_group, nullptr)),
      miss_prog_group(std::exchange(other.miss_prog_group, nullptr)),
      hitgroup_prog_group(std::exchange(other.hitgroup_prog_group, nullptr)),
      module(std::exchange(other.module, nullptr)),
      sbt(std::exchange(other.sbt, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}

FindTetrahedraPipeline::~FindTetrahedraPipeline() noexcept(false) {
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (sbt.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
    if (sbt.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
    sbt = {};
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}

void FindTetrahedraPipeline::find_tetrahedra(const TetrahedraStructure *tetrahedra_structure,
                                             const size_t num_points,
                                             const float3 *points,
                                             unsigned int *tetrahedra_out,
                                             float3 *barycentric_coordinates_out,
                                             uint4 *vertex_indices_out) {
    CUDA_CHECK(cudaSetDevice(device));
    float2 *triangle_barycentric_coordinates;
    unsigned int *shared_visited_triangles;
    float *triangle_hit_distances;

    ParamsFindTetrahedra params;
    params.barycentric_coordinates = barycentric_coordinates_out;
    params.ray_origins = points;
    params.tetrahedra = tetrahedra_out;
    params.vertex_indices = vertex_indices_out;
    params.triangle_indices = tetrahedra_structure->triangle_indices();
    params.triangle_tetrahedra = tetrahedra_structure->triangle_tetrahedra();
    params.handle = tetrahedra_structure->gas_handle();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &params, sizeof(params),
        cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt, num_points, 1, 1));
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TraceRaysTrianglesPipeline::TraceRaysTrianglesPipeline(const OptixDeviceContext &context, int8_t device) : device(device), context(context) {
    // Initialize fields
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Create module
    //
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

// The following is not supported in Optix 7.2
// #define XSTR(x) STR(x)
// #define STR(x) #x
// #pragma message "Optix ABI version is: " XSTR(OPTIX_ABI_VERSION)
#if (OPTIX_ABI_VERSION > 54)
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 1;
        // https://forums.developer.nvidia.com/t/how-to-calculate-numattributevalues-of-optixpipelinecompileoptions/110833
        pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        std::string input = TraceRaysTrianglesPipeline::load_ptx_data();
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input.c_str(),
            input.size(),
            log,
            &sizeof_log,
            &module));
    }

    //
    // Create program groups
    //
    {
        OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};  //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ms";
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0,  // maxCCDepth
                                               0,  // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1  // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size,
            cudaMemcpyHostToDevice));

        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(ParamsTraceRaysTriangles)));
}

void TraceRaysTrianglesPipeline::trace_rays(const TetrahedraStructure *tetrahedra_structure,
                                            const size_t num_rays,
                                            const unsigned int max_ray_triangles,
                                            const float3 *ray_origins,
                                            const float3 *ray_directions,
                                            unsigned int *num_visited_triangles_out,
                                            unsigned int *visited_triangles_out,
                                            float2 *barycentric_coordinates_out,
                                            float *hit_distances_out,
                                            uint3 *vertex_indices_out) {
    ParamsTraceRaysTriangles params;
    params.triangle_tetrahedra = tetrahedra_structure->triangle_tetrahedra();
    params.triangle_indices = tetrahedra_structure->triangle_indices();
    params.num_visited_triangles = num_visited_triangles_out;
    params.visited_triangles = visited_triangles_out;
    params.vertex_indices = vertex_indices_out;
    params.max_ray_triangles = max_ray_triangles;
    params.barycentric_coordinates = barycentric_coordinates_out;
    params.hit_distances = hit_distances_out;
    params.handle = tetrahedra_structure->gas_handle();
    params.ray_origins = ray_origins;
    params.ray_directions = ray_directions;

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &params, sizeof(params),
        cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt, num_rays, 1, 1));
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TraceRaysTrianglesPipeline::TraceRaysTrianglesPipeline(TraceRaysTrianglesPipeline &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      pipeline(std::exchange(other.pipeline, nullptr)),
      raygen_prog_group(std::exchange(other.raygen_prog_group, nullptr)),
      miss_prog_group(std::exchange(other.miss_prog_group, nullptr)),
      hitgroup_prog_group(std::exchange(other.hitgroup_prog_group, nullptr)),
      module(std::exchange(other.module, nullptr)),
      sbt(std::exchange(other.sbt, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}

TraceRaysTrianglesPipeline::~TraceRaysTrianglesPipeline() noexcept(false) {
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device));
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (sbt.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
    if (sbt.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
    sbt = {};
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}

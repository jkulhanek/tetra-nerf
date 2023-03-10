#include "tetrahedra_tracer.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "optix_types.h"
#include "utils/exception.h"
#include "utils/vec_math.h"

namespace fs = std::filesystem;

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

void TetrahedraTracer::load_tetrahedra(const size_t num_vertices,
                                       const size_t num_cells,
                                       const float3 *d_vertices,
                                       const uint4 *cells) {
    const size_t block_size = 1024;

    uint3 *triangle_indices; 
    size_t num_triangles = num_cells * 4;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&triangle_indices),
        num_triangles * sizeof(uint3)
    ));
    convert_cells_to_vertices(block_size, num_cells, cells, triangle_indices);

    // Release if previous structure exists
    release_accel_structure();

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
    triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(triangle_indices);
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
        &gas_handle,
        nullptr,  // emitted property list
        0         // num emitted properties
        ));
    this->num_vertices = num_vertices;
    this->num_cells = num_cells;
    this->tetrahedra_vertices = d_vertices;
    this->tetrahedra_cells = cells;

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(triangle_indices)));
}

std::string TetrahedraTracer::load_ptx_data() {
    return std::string((char *)ptx_code_file);
}

TetrahedraTracer::TetrahedraTracer(int8_t device) {
    this->device = device;
    // Initialize fields
    context = nullptr;
    d_gas_output_buffer = 0;
    gas_handle = 0;
    raygen_prog_group = nullptr;
    miss_prog_group = nullptr;
    hitgroup_prog_group = nullptr;
    pipeline = nullptr;
    sbt = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};

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

    //
    // Create module
    //
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

// The following is not supported in Optix 7.2
#if(OPTIX_ABI_VERSION > 67)
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

        std::string input = TetrahedraTracer::load_ptx_data();
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

void TetrahedraTracer::release_accel_structure() {
    CUDA_CHECK(cudaSetDevice(device));
    if (d_gas_output_buffer != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));
        d_gas_output_buffer = 0;
    }
    if (d_gas_output_buffer != 0) {
        gas_handle = 0;
    }

    this->tetrahedra_vertices = nullptr;
    this->tetrahedra_cells = nullptr;
}

TetrahedraTracer::~TetrahedraTracer() noexcept(false) {
    CUDA_CHECK(cudaSetDevice(device));
    release_accel_structure();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));

    OPTIX_CHECK(optixDeviceContextDestroy(context));
}

void TetrahedraTracer::trace_rays(const size_t num_rays,
                                  const unsigned int max_ray_triangles,
                                  const unsigned int max_visited_cells,
                                  const float3 *ray_origins,
                                  const float3 *ray_directions,
                                  unsigned int *num_visited_cells_out,
                                  unsigned int *visited_cells_out,
                                  float4 *barycentric_coordinates_out,
                                  float *hit_distances_out) {

    CUDA_CHECK(cudaSetDevice(device));
    float2* triangle_barycentric_coordinates;
    unsigned int* shared_visited_triangles;
    float* triangle_hit_distances;

    // TODO: we can reuse the triangle memory
    // No need to allocate again here
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&triangle_barycentric_coordinates),
        sizeof(float2) * max_ray_triangles * num_rays
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&shared_visited_triangles),
        sizeof(unsigned int) * (1+max_ray_triangles) * num_rays
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&triangle_hit_distances),
        sizeof(float) * max_ray_triangles * num_rays
    ));
    unsigned int* num_visited_triangles = shared_visited_triangles;
    unsigned int* visited_triangles = shared_visited_triangles + num_rays;

    {
        Params params;
        params.shared_visited_triangles = shared_visited_triangles;
        params.max_ray_triangles = max_ray_triangles;
        params.max_visited_cells = max_visited_cells;
        params.barycentric_coordinates = triangle_barycentric_coordinates;
        params.hit_distances = triangle_hit_distances;
        params.handle = gas_handle;
        params.ray_origins = ray_origins;
        params.ray_directions = ray_directions;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, num_rays, 1, 1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    post_process_tetrahedra(
        num_rays,
        max_ray_triangles,
        max_visited_cells,
        num_visited_triangles,
        visited_triangles,
        triangle_barycentric_coordinates,
        triangle_hit_distances,
        visited_cells_out,
        num_visited_cells_out,
        barycentric_coordinates_out,
        hit_distances_out,
        eps);


    CUDA_CHECK(cudaFree(triangle_barycentric_coordinates));
    CUDA_CHECK(cudaFree(triangle_hit_distances));
    CUDA_CHECK(cudaFree(shared_visited_triangles));
}

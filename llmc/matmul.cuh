/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// GELU can be either fused (cublasLt) or non-fused (gelu.h)
#include "gelu.cuh"

#include "triton_matmul/triton_matmul_kernels.h"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    // Strided Batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

// NOTE: matmul_triton computes A @ B row column major layout, which means matmul_triton and matmul_cublaslt can not be used interchangably!
void matmul_triton(floatX *d, const floatX *a, const floatX *b, const floatX *bias,
                   int m, int n, int k, cudaStream_t stream = nullptr, bool transA = true, bool transB = false,
                   int batch_count = 0, size_t strideA = 0, size_t strideB = 0, size_t strideOut = 0,
                   bool accumulate = false, floatX *pre_gelu = nullptr, bool backward = false) {

    // TODO: Not yet supported
    assert(batch_count == 0);

    static bool kernels_loaded = false;

    static CUmodule cuda_modules[NUM_MATMUL_KERNEL_PERMUTATIONS] = {};
    static CUfunction cuda_functions[NUM_MATMUL_KERNEL_PERMUTATIONS] = {};

    // load kernels on first invocation
    if (!kernels_loaded) {
        for (int i = 0; i < NUM_MATMUL_KERNEL_PERMUTATIONS; i++) {
            const std::string &kernel_source = TRITON_MATMUL_KERNEL_PTX_SOURCES[i];
            if (CUresult status
                        = cuModuleLoadDataEx(cuda_modules + i, kernel_source.c_str(), 0, nullptr, nullptr);
                    status != CUDA_SUCCESS) {
                auto error = (cudaError_t) status;
                printf("%d\n", error);
                return;
            }
        }

        for (int i = 0; i < NUM_MATMUL_KERNEL_PERMUTATIONS; i++) {
            const std::string &kernel_name = TRITON_MATMUL_KERNEL_FUNCTION_NAMES[i];
            CUfunction *function = cuda_functions + i;
            if (CUresult status
                        = cuModuleGetFunction(function, cuda_modules[i], kernel_name.c_str());
                    status != CUDA_SUCCESS) {
                auto error = (cudaError_t) status;
                printf("%d\n", error);
                return;
            }

            // set dynamic shared memory if necessary
            if (TRITON_MATMUL_KERNEL_SHARED_MEMORY_SIZES[i] >= 49152) {
                CUdevice device{};
                cuDeviceGet(&device, 0); // TODO: HACK

                int shared_optin{};
                cudaCheck(cuDeviceGetAttribute(&shared_optin,
                                                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                                device));
                if (shared_optin >= 49152) {
                    cudaCheck(cuFuncSetCacheConfig(*function, CU_FUNC_CACHE_PREFER_SHARED));

                    int shared_total{}, shared_static{};
                    cudaCheck(cuDeviceGetAttribute(&shared_total,
                                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));

                    cudaCheck(cuFuncGetAttribute(
                            &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *function));

                    cudaCheck(cuFuncSetAttribute(*function,
                                                  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                  shared_optin - shared_static));
                }
            }
        }
        kernels_loaded = true;
    }
    bool use_bias = bias != nullptr;
    bool fuse_gelu = pre_gelu != nullptr;

    int kernel_idx = static_cast<int>(accumulate) + (2 * static_cast<int>(transA)) + (4 * static_cast<int>(transB)) +
                     (8 * static_cast<int>(use_bias)) + (16 * static_cast<int>(fuse_gelu)) + (16 * static_cast<int>(backward && fuse_gelu));

    CUfunction kernel_function = cuda_functions[kernel_idx];

    int block_size_m = TRITON_MATMUL_KERNEL_BLOCK_SIZES_M[kernel_idx];
    int block_size_n = TRITON_MATMUL_KERNEL_BLOCK_SIZES_N[kernel_idx];

    int num_warps = TRITON_MATMUL_KERNEL_NUM_WARPS[kernel_idx];
    int shared_memory_size = TRITON_MATMUL_KERNEL_SHARED_MEMORY_SIZES[kernel_idx];

    // define grid
    unsigned int gridX = ((m + block_size_m - 1) / block_size_m) *
                         ((n + block_size_n - 1) / block_size_n);

    auto stride_am = transA ? 1 : k;
    auto stride_ak = transA ? m : 1;

    auto stride_bk = transB ? 1 : n;
    auto stride_bn = transB ? k : 1;

    auto stride_cm = n;
    auto stride_cn = 1;

    // signature:
    // u64 a_ptr, u64 b_ptr, u64 c_ptr,
    // <opt: u64 bias_ptr>, <opt: u64 aux_ptr>, 
    // u32 m, u32 n, u32 k,
    // u32 stride_am, u32 stride_ak,
    // u32 stride_bk, u32 stride_bn,
    // u32 stride_cm, u32 stride_cn

    int *stride_params[6] = {
            &stride_am, &stride_ak,
            &stride_bk, &stride_bn,
            &stride_cm, &stride_cn
    };
    std::vector<void *> parameters{
            &a, &b, &d
    };

    // NOTE: Triton automatically inlines parameters that are 0 or 1, even though they are not constexpr!
    // Therefore we have to dynamically parameters them depending on the kernel permutation
    if (bias != nullptr) {
        parameters.push_back(&bias);
    }

    if (pre_gelu != nullptr) {
        parameters.push_back(&pre_gelu);
    }

    parameters.push_back(&m);
    parameters.push_back(&n);
    parameters.push_back(&k);

    // Handling of inlining of stride parameters that are one
    int n_inlined = 0;
    for (int *stride: stride_params) {
        if (*stride != 1) {
            parameters.push_back(stride);
            n_inlined++;
        }
    }
    assert(n_inlined == 3);
    cudaCheck(cuLaunchKernel(kernel_function,
                   gridX, 1, 1,
                   32 * num_warps, 1, 1,
                   shared_memory_size,
                   stream,
                   parameters.data(),
                   nullptr));
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1, bool use_lp_accumulator=true) {

    // Weirdness warning: cuBLASLt expects column-major layout, but the rest of the code uses row-major layout!
    // logical shapes:
    // weight: (OC, C)
    // inp: (B*T, C)

    // memory layout = column-major, hence physical shapes:
    // weight: (C, OC)
    // inp: (C, B*T)

    // (C, OC).T @ (C, B*T) = (OC, B*T) in column-major

    // comumn-major output is misinterpreted as row-major -> invisible transpose
    // (OC, B*T) -> (B*T, OC)


    // By default only fuse GELU for H100+ as cuBLAS seems to be inefficient for fused GELU on Ada/Ampere (?)
    if (gelu_fusion < 1 && pre_gelu) {
        if (PRECISION_MODE == PRECISION_FP16) {
            matmul_triton(pre_gelu, inp, weight, bias, B*T, OC, C, stream, false, true, 0, 0, 0, 0, false, NULL, false);    
        } else {
            matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        }
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        if (PRECISION_MODE == PRECISION_FP16 && use_lp_accumulator) {
            matmul_triton(out, inp, weight, bias, B*T, OC, C, stream, false, true, 0, 0, 0, 0, false, pre_gelu, false);    
        } else {
            matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
        }
    }
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1, bool use_lp_accumulator=true) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    if (PRECISION_MODE == PRECISION_FP16 && use_lp_accumulator && false) { // only respect use_lp_accumulator when OC is the inner dimension where the accumulator matters (eg. vocab size, which is really large and will overflow!)
        // logical shapes:
        // weight: (OC, C)
        // dout: (B*T, OC)
        // memory layout = row-major, hence physical identical;
        // (B*T, OC) @ (OC, C) = (B*T, C)
        // A = (m, k), B = (k, n), C = (m, n)
        // hence: m=B*T, k=OC, n=C

        matmul_triton(dinp, dout, weight, NULL, B*T, C, OC, stream, false, false, 0, 0, 0, 0, false,
                    pre_gelu, true);
    } else {
        // logical shapes:
        // weight: (OC, C)
        // dout: (B*T, OC)

        // memory layout = column-major, hence physical shapes:
        // weight: (C, OC)
        // dout: (OC, B*T)

        // (C, OC) @ (OC, B*T) = (C, B*T) in column-major

        // comumn-major output is misinterpreted as row-major -> invisible transpose
        // (C, B*T) -> (B*T, C)

        matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                        gelu_fusion >= 2 ? pre_gelu : NULL, true);

        // backward GELU (if it wasn't fused into the matmul above)
        if (gelu_fusion < 2 && pre_gelu) {
            gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
        }
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one

    if (PRECISION_MODE != PRECISION_FP16) {
        // logical shapes:
        // inp: (B*T, C)
        // dout: (B*T, OC)

        // memory layout = column-major, hence physical shapes:
        // inp: (C, B*T)
        // dout: (OC, B*T)

        // (C, B*T) @ (OC, B*T).T = (C, OC) in column-major
        // comumn-major output is misinterpreted as row-major -> invisible transpose
        // (C, OC) -> (OC, C)
        matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                        true /* accumulate */, NULL, true);
    } else {
        // logical shapes:
        // inp: (B*T, C)
        // dout: (B*T, OC)
        // memory layout = row-major, hence physical identical;
        // (B*T, OC).T @ (B*T, C) = (OC, C)
        // A = (m, k), B = (k, n), C = (m, n)
        // hence: m=OC, k=B*T, n=C
        matmul_triton(dweight, dout, inp, NULL, OC, C, B*T, stream, true, false, 0, 0, 0, 0, true, NULL, true);
    }

}

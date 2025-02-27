
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <CL/sycl.hpp>
#include <cute/tensor.hpp>
#include <syclcompat.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif


template <class MShape, class NShape, class KShape,
        class TA, class AStride, class ABlockLayout, class AThreadLayout,
        class TB, class BStride, class BBlockLayout, class BThreadLayout,
        class TC, class CStride, class CBlockLayout, class CThreadLayout,
        class Alpha, class Beta>
void gemm_device(MShape M, NShape N, KShape K,
                 TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
                 TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
                 TC      * C, CStride dC, CBlockLayout       , CThreadLayout tC,
                 Alpha alpha, Beta beta)
{
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    //CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
    //CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));      // BLK_N
    CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));        // BLK_K

    // Shared memory buffers
    auto smemA = syclcompat::local_mem<TA[cosize_v<ABlockLayout>]>();
    auto smemB = syclcompat::local_mem<TB[cosize_v<BBlockLayout>]>();
    auto sA = make_tensor(make_smem_ptr(smemA), blockA);               // (BLK_M,BLK_K)
    auto sB = make_tensor(make_smem_ptr(smemB), blockB);               // (BLK_N,BLK_K)

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

    // Get the appropriate blocks for this thread block --
    // potential for thread block locality
    auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));// (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(syclcompat::work_group_id::x(), syclcompat::work_group_id::y(), _);            // (m,n,k)

    auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    //
    // Partition the copying of A and B tiles across the threads
    //

    // TUTORIAL: Example of simple partitioning of A|B tiles over tA|tB
    //   Default is a raked partition, but can be changed with Step<X,Y> parameter

    auto tAgA = local_partition(gA, tA, syclcompat::local_id::x());                  // (THR_M,THR_K,k)
    auto tAsA = local_partition(sA, tA, syclcompat::local_id::x());                  // (THR_M,THR_K)

    auto tBgB = local_partition(gB, tB, syclcompat::local_id::x());                  // (THR_N,THR_K,k)
    auto tBsB = local_partition(sB, tB, syclcompat::local_id::x());                  // (THR_N,THR_K)

    //
    // Define C accumulators and A/B partitioning
    //

    // TUTORIAL: Example of partitioning via projections of tC

    // Partition sA (M,K) by the rows of tC
    auto tCsA = local_partition(sA, tC, syclcompat::local_id::x(), Step<_1, X>{});   // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    auto tCsB = local_partition(sB, tC, syclcompat::local_id::x(), Step< X,_1>{});   // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC = local_partition(gC, tC, syclcompat::local_id::x(), Step<_1,_1>{});   // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);                              // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

#if 0
    if(thread0()) {
    print("mA\n");
    print(mA.shape()); print("\n"); print(mA.stride());
    print("\n\ngA\n");
    print(gA.shape()); print("\n"); print(gA.stride());
    print("\n\ntAgA\n");
    print(tAgA.shape()); print("\n"); print(tAgA.stride());
    print("\n\nsA\n");
    print(sA.shape()); print("\n"); print(sA.stride());
    print("\n\ntAsA\n");
    print(tAsA.shape()); print("\n"); print(tAsA.stride());
    print("\n\n");
  }
#endif

#if 0
    if(thread0()) {
    print("mB\n");
    print(mB.shape()); print("\n"); print(mB.stride());
    print("\n\ngB\n");
    print(gB.shape()); print("\n"); print(gB.stride());
    print("\n\ntBgB\n");
    print(tBgB.shape()); print("\n"); print(tBgB.stride());
    print("\n\nsB\n");
    print(sB.shape()); print("\n"); print(sB.stride());
    print("\n\ntBsB\n");
    print(tBsB.shape()); print("\n"); print(tBsB.stride());
    print("\n\n");
  }
#endif

#if 0
    if(thread0()) {
    print("mC\n");
    print(mC.shape()); print("\n"); print(mC.stride());
    print("\n\ngC\n");
    print(gC.shape()); print("\n"); print(gC.stride());
    print("\n\ntCsA\n");
    print(tCsA.shape()); print("\n"); print(tCsA.stride());
    print("\n\ntCsB\n");
    print(tCsB.shape()); print("\n"); print(tCsB.stride());
    print("\n\ntCgC\n");
    print(tCgC.shape()); print("\n"); print(tCgC.stride());
    print("\n\ntCrC\n");
    print(tCrC.shape()); print("\n"); print(tCrC.stride());
    print("\n\n");
  }
#endif

#if 1

    // TUTORIAL: Example of a very simple compute loop
    //   Data is read from global to shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared memory directly via the tC partitioning

    auto k_max = size<2>(tAgA);

    for (int k = 0; k < k_max; ++k)
    {
        // Copy gmem to smem
        copy(tAgA(_,_,k), tAsA);
        copy(tBgB(_,_,k), tBsB);

        // In case copy uses cp.async, make sure that the cp.async
        // instructions are ordered with respect to other cp.async
        // instructions (fence), then wait on all the outstanding copy
        // operations (wait<0>()).  __syncthreads() alone does not do
        // this.
        //
        // NOTE: cp_async_wait<0>() currently issues cp.async.wait_all.
        // This is equivalent to cp.async.commit_group followed by
        // cp.async_wait_group 0.  This should make the first
        // cp_async_fence() (which also issues cp.async.commit_group)
        // redundant.  The tutorial works as-is, so we'll leave the
        // redundant fence in for now and study its removal later.
        cp_async_fence();
        cp_async_wait<0>();

        syclcompat::wg_barrier();

        // Compute gemm on smem
        gemm(tCsA, tCsB, tCrC);

        syclcompat::wg_barrier();
    }

#endif

    //
    // Epilogue
    //

    axpby(alpha, tCrC, beta, tCgC);
}


template <typename TA, typename TB, typename TC,
        typename Alpha, typename Beta>
void
gemm(sycl::queue q, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    // Define strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    // Define block sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};

    // Define the block layouts (static)
    auto sA = make_layout(make_shape(bM,bK));
    auto sB = make_layout(make_shape(bN,bK));
    auto sC = make_layout(make_shape(bM,bN));

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    const auto block = syclcompat::dim3(size(tC));
    const auto grid = syclcompat::dim3(ceil_div(size(M), size(bM)),
                                       ceil_div(size(N), size(bN)));

    syclcompat::launch<
            gemm_device<
                    int, int, int,
                    TA, decltype(dA), decltype(sA), decltype(tA),
                    TB, decltype(dB), decltype(sB), decltype(tB),
                    TC, decltype(dC), decltype(sC), decltype(tC),
                    Alpha, Beta
            >
    >(grid, block, q, M,  N,  K,
      A, dA, sA, tA,
      B, dB, sB, tB,
      C, dC, sC, tC,
      alpha, beta);
}

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>

void test_gemm(int m, int n, int k)
{
    auto q = sycl::queue { sycl::gpu_selector_v } ;

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    auto d_A = sycl::malloc_device<TA>(m*k, q);
    auto d_B = sycl::malloc_device<TB>(n*k, q);
    auto d_C = sycl::malloc_device<TC>(m*n, q);

    q.memcpy(d_A, h_A.data(), m*k * sizeof(TA)).wait();
    q.memcpy(d_B, h_B.data(), n*k * sizeof(TB)).wait();
    q.memcpy(d_C, h_C.data(), m*n * sizeof(TC)).wait();

    TI alpha = 1.0;
    TI beta  = 0.0;

    double tflops = (2.0*m*n*k) * 1e-12;

    const int timing_iterations = 100;
    GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
    //
    // cuBLas
    //

    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<TA> dc_A = h_A;
    thrust::device_vector<TB> dc_B = h_B;
    thrust::device_vector<TC> dc_C = h_C;

    // Run once
    blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       m, n, k,
                       &alpha,
                       dc_A.data().get(), m,
                       dc_B.data().get(), n,
                       &beta,
                       dc_C.data().get(), m);
    CUTE_CHECK_LAST();

    thrust::host_vector<TC> cublas_result = dc_C;

    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, n, k,
                           &alpha,
                           dc_A.data().get(), m,
                           dc_B.data().get(), n,
                           &beta,
                           dc_C.data().get(), m);
    }
    double cublas_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUBLAS_GEMM:   [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cublas_time, cublas_time*1000);

#else

    std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
    "If you would like to enable verification with cuBLAS, "
    "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
    "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS

    //
    // CuTe
    //

    // Run once (and check)

    gemm(q, m, n, k,
         alpha,
         d_A, m,
         d_B, n,
         beta,
         d_C, m);
    CUTE_CHECK_LAST();
    q.wait_and_throw();

    // Timing iterations

    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(q, m, n, k,
             alpha,
             d_A, m,
             d_B, n,
             beta,
             d_C, m);
    }

    q.wait();

    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("SYCL_CUTE_GEMM:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

    std::vector<TC> cute_result(m*n);
    q.memcpy(cute_result.data(), d_C, m*n * sizeof(TC)).wait();

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
    printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

    auto host_matrix_to_const_column_major_cute_tensor =
            [](const auto& X, int num_rows, int num_cols, int LDX) {
                const auto shape = cute::Shape<int, int>{num_rows, num_cols};
                const auto strides = cute::Stride<int, int>{1, LDX};
                return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
            };

    const auto A_view = host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
    // B^T is k x n, so B is n x k.
    const auto B_view = host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
    const auto C_computed_view = host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
    const auto C_expected_view = host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
    print_matrix_multiply_mollified_relative_error("float", A_view, B_view, C_computed_view, C_expected_view);

#endif // CUTLASS_ENABLE_CUBLAS
}


int main(int argc, char** argv)
{
    int m = 5120;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 5120;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    test_gemm(m, n, k);

    return 0;
}

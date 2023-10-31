/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <cute/tensor.hpp>

#include <chrono>
using test_clock = std::chrono::high_resolution_clock;

#include "cutlass/util/print_error.hpp"
// #include "cutlass/util/GPU_Clock.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
// #  include "cutlass/util/cublas_wrappers.hpp"
#  include "cutlass/util/dpct_output/cublas_wrappers.hpp"
#endif
// #include "cutlass/util/helper_cuda.hpp"

template <class MShape, class NShape, class KShape, class TA, class AStride,
          class ABlockLayout, class AThreadLayout, class TB, class BStride,
          class BBlockLayout, class BThreadLayout, class TC, class CStride,
          class CBlockLayout, class CThreadLayout, class Alpha, class Beta>
/*
DPCT1110:2: The total declared local variable size in device function
gemm_device exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
static

    void
    gemm_device(MShape M, NShape N, KShape K, TA const *A, AStride dA,
                ABlockLayout blockA, AThreadLayout tA, TB const *B, BStride dB,
                BBlockLayout blockB, BThreadLayout tB, TC *C, CStride dC,
                CBlockLayout, CThreadLayout tC, Alpha alpha, Beta beta,
                const sycl::nd_item<3> &item_ct1, TA *smemA, TB *smemB)
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

  auto sA = make_tensor(make_smem_ptr(smemA), blockA);               // (BLK_M,BLK_K)
  auto sB = make_tensor(make_smem_ptr(smemB), blockB);               // (BLK_N,BLK_K)

  // Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block --
  // potential for thread block locality
  auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));// (BLK_M,BLK_N,BLK_K)
  auto blk_coord =
      make_coord(item_ct1.get_group(2), item_ct1.get_group(1), _); // (m,n,k)

  auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple partitioning of A|B tiles over tA|tB
  //   Default is a raked partition, but can be changed with Step<X,Y> parameter

  auto tAgA =
      local_partition(gA, tA, item_ct1.get_local_id(2)); // (THR_M,THR_K,k)
  auto tAsA =
      local_partition(sA, tA, item_ct1.get_local_id(2)); // (THR_M,THR_K)

  auto tBgB =
      local_partition(gB, tB, item_ct1.get_local_id(2)); // (THR_N,THR_K,k)
  auto tBsB =
      local_partition(sB, tB, item_ct1.get_local_id(2)); // (THR_N,THR_K)

  //
  // Define C accumulators and A/B partitioning
  //

  // TUTORIAL: Example of partitioning via projections of tC

  // Partition sA (M,K) by the rows of tC
  auto tCsA = local_partition(sA, tC, item_ct1.get_local_id(2),
                              Step<_1, X>{}); // (THR_M,BLK_K)
  // Partition sB (N,K) by the cols of tC
  auto tCsB = local_partition(sB, tC, item_ct1.get_local_id(2),
                              Step<X, _1>{}); // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  auto tCgC = local_partition(gC, tC, item_ct1.get_local_id(2),
                              Step<_1, _1>{}); // (THR_M,THR_N)

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

    /*
    DPCT1118:0: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Compute gemm on smem
    gemm(tCsA, tCsB, tCrC);

    /*
    DPCT1118:1: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
void gemm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B,
          int ldB, Beta beta, TC *C, int ldC,
          dpct::queue_ptr stream = &dpct::get_in_order_queue())
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

  sycl::range<3> dimBlock(1, 1, size(tC));
  sycl::range<3> dimGrid(1, ceil_div(size(N), size(bN)),
                         ceil_div(size(M), size(bM)));
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<TA, 1> smemA_acc_ct1(
        sycl::range<1>(cosize_v<decltype(sA)>), cgh);
    sycl::local_accessor<TB, 1> smemB_acc_ct1(
        sycl::range<1>(cosize_v<decltype(sB)>), cgh);

    cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                       gemm_device(M, N, K, A, dA, sA, tA, B, dB, sB, tB, C, dC,
                                   sC, tC, alpha, beta, item_ct1,
                                   (TA *)smemA_acc_ct1.get_pointer(),
                                   (TB *)smemB_acc_ct1.get_pointer());
                     });
  }).wait();
}

#include <cstdlib>
#include <cstdio>
#include <cassert>

void test_gemm(int m, int n, int k)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  dpct::device_info deviceProp;
  dpct::dev_mgr::instance().get_device(0).get_device_info(
      deviceProp);
  std::cout << "Running on device: "
            << deviceProp.get_name() << "\n";
  //cute::device_init(0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  std::vector<TA> h_A(m * k);
  std::vector<TB> h_B(n * k);
  std::vector<TC> h_C(m * n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  dpct::device_vector<TA> d_A = h_A;
  dpct::device_vector<TB> d_B = h_B;
  dpct::device_vector<TC> d_C = h_C;

  TI alpha = 1.0;
  TI beta  = 0.0;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 16;
  //GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //
  dpct::queue_ptr handle = &dpct::get_in_order_queue();
  //cublasCreate(&handle);

  // Run once
  d_C = h_C;
  blam::cublas::gemm(handle, oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                     m, n, k,
                     &alpha,
                     d_A.data(), m,
                     d_B.data(), n,
                     &beta,
                     d_C.data(), m);
  /*
  DPCT1010:5: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:6: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1001:3: The statement could not be removed.
  */
  /*
  DPCT1000:4: Error handling if-stmt was detected but could not be rewritten.
  */
  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUTE_CHECK_LAST())*/
      ;

  std::vector<TC> cublas_result = d_C;

  // Timing iterations
  //timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    blam::cublas::gemm(handle, oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                       m, n, k,
                       &alpha,
                       d_A.data(), m,
                       d_B.data(), n,
                       &beta,
                       d_C.data(), m);
  }
  //double cublas_time = timer.seconds() / timing_iterations;
  /*
  DPCT1010:9: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:10: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1001:7: The statement could not be removed.
  */
  /*
  DPCT1000:8: Error handling if-stmt was detected but could not be rewritten.
  */
  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUTE_CHECK_LAST())*/
      ;
  // printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);

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
  d_C = h_C;
  gemm(m, n, k,
       alpha,
       d_A.data(), m,
       d_B.data(), n,
       beta,
       d_C.data(), m);

  std::vector<TC> cute_result = d_C;

  // Timing iterations
  //timer.start();
  auto start = test_clock::now();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(m, n, k,
         alpha,
         d_A.data(), m,
         d_B.data(), n,
         beta,
         d_C.data(), m);
  }
  dpct::get_in_order_queue().wait();
  auto end = test_clock::now();
  std::chrono::duration<double> delta = end - start;
  double cute_time = delta.count() / timing_iterations;
  //double cute_time = timer.seconds() / timing_iterations;

  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

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

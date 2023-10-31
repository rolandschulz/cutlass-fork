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

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <complex>

//-- BLAM_DEBUG_OUT ---------------------------------------------------------
#ifdef BLAM_DEBUG
# include <iostream>
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)    std::cerr << "BLAM: " << msg << std::endl
#  define BLAM_DEBUG_OUT_2(msg)  std::cerr << msg << std::endl
# endif // BLAM_DEBUG_OUT
#else
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)
#  define BLAM_DEBUG_OUT_2(msg)
# endif // BLAM_DEBUG_OUT
#endif // BLAM_DEBUG

// User could potentially define ComplexFloat/ComplexDouble instead of std::
#ifndef BLAM_COMPLEX_TYPES
#define BLAM_COMPLEX_TYPES 1
namespace blam {
template <typename T> using Complex = std::complex<T>;
using ComplexFloat = std::complex<float>;
using ComplexDouble = std::complex<double>;
}
#endif // BLAM_COMPLEX_TYPES

// User could potentially define Half instead of cute::
#ifndef BLAM_HALF_TYPE
#define BLAM_HALF_TYPE 1
#include <cute/numeric/half.hpp>
#include <dpct/lib_common_utils.hpp>

namespace blam {
using Half = cute::half_t;
}
#endif // BLAM_HALF_TYPE

namespace blam
{
namespace cublas
{

inline const char *cublas_get_error(int status)
{
  switch (status) {
    case 0:
      return "CUBLAS_STATUS_SUCCESS";
    case 1:
      return "CUBLAS_STATUS_NOT_INITIALIZED -- The cuBLAS library was not initialized.";
    case 3:
      return "CUBLAS_STATUS_ALLOC_FAILED -- Resource allocation failed inside the cuBLAS library.";
    case 7:
      return "CUBLAS_STATUS_INVALID_VALUE -- An unsupported value or parameter was passed to the function.";
    case 8:
      return "CUBLAS_STATUS_ARCH_MISMATCH -- The function requires a feature absent from the device architecture.";
    case 11:
      return "CUBLAS_STATUS_MAPPING_ERROR -- An access to GPU memory space failed.";
    case 13:
      return "CUBLAS_STATUS_EXECUTION_FAILED -- The GPU program failed to execute.";
    case 14:
      return "CUBLAS_STATUS_INTERNAL_ERROR -- An internal cuBLAS operation failed.";
    case 15:
      return "CUBLAS_STATUS_NOT_SUPPORTED -- The functionality requested is not supported.";
    case 16:
      return "CUBLAS_STATUS_LICENSE_ERROR -- An error was detected when checking the current licensing.";
    default:
      return "CUBLAS_ERROR -- <unknown>";
  }
}

inline bool cublas_is_error(int status)
{
  return status != 0;
}


// hgemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const Half *alpha, const Half *A, int ldA, const Half *B,
                int ldB, const Half *beta, Half *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasHgemm");

  return DPCT_CHECK_ERROR(dpct::gemm(
      *handle, transA, transB, m, n, k,
      reinterpret_cast<const sycl::half *>(alpha),
      reinterpret_cast<const sycl::half *>(A), dpct::library_data_t::real_half,
      ldA, reinterpret_cast<const sycl::half *>(B),
      dpct::library_data_t::real_half, ldB,
      reinterpret_cast<const sycl::half *>(beta),
      reinterpret_cast<sycl::half *>(C), dpct::library_data_t::real_half, ldC,
      dpct::library_data_t::real_half));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// mixed hf gemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const float *alpha, const Half *A, int ldA, const Half *B,
                int ldB, const float *beta, float *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasGemmEx mixed half-float");

  return DPCT_CHECK_ERROR(dpct::gemm(
      *handle, transA, transB, m, n, k, alpha,
      reinterpret_cast<const sycl::half *>(A), dpct::library_data_t::real_half,
      ldA, reinterpret_cast<const sycl::half *>(B),
      dpct::library_data_t::real_half, ldB, beta, C,
      dpct::library_data_t::real_float, ldC, dpct::library_data_t::real_float));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// igemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const int32_t *alpha, const int8_t *A, int ldA, const int8_t *B,
                int ldB, const int32_t *beta, int32_t *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasIgemm");

  return DPCT_CHECK_ERROR(dpct::gemm(*handle, transA, transB, m, n, k, alpha, A,
                                     dpct::library_data_t::real_int8, ldA, B,
                                     dpct::library_data_t::real_int8, ldB, beta,
                                     C, dpct::library_data_t::real_int32, ldC,
                                     dpct::library_data_t::real_int32));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// sgemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const float *alpha, const float *A, int ldA, const float *B,
                int ldB, const float *beta, float *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasSgemm");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
      *handle, transA, transB, m, n, k, dpct::get_value(alpha, *handle), A, ldA,
      B, ldB, dpct::get_value(beta, *handle), C, ldC));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// dgemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const double *alpha, const double *A, int ldA, const double *B,
                int ldB, const double *beta, double *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasDgemm");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
      *handle, transA, transB, m, n, k, dpct::get_value(alpha, *handle), A, ldA,
      B, ldB, dpct::get_value(beta, *handle), C, ldC));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cgemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const ComplexFloat *alpha, const ComplexFloat *A, int ldA,
                const ComplexFloat *B, int ldB, const ComplexFloat *beta,
                ComplexFloat *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasCgemm");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
      *handle, transA, transB, m, n, k,
      dpct::get_value(reinterpret_cast<const sycl::float2 *>(alpha), *handle),
      (std::complex<float> *)reinterpret_cast<const sycl::float2 *>(A), ldA,
      (std::complex<float> *)reinterpret_cast<const sycl::float2 *>(B), ldB,
      dpct::get_value(reinterpret_cast<const sycl::float2 *>(beta), *handle),
      (std::complex<float> *)reinterpret_cast<sycl::float2 *>(C), ldC));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// zgemm
inline int gemm(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, int m, int n, int k,
                const ComplexDouble *alpha, const ComplexDouble *A, int ldA,
                const ComplexDouble *B, int ldB, const ComplexDouble *beta,
                ComplexDouble *C, int ldC) try {
  BLAM_DEBUG_OUT("cublasZgemm");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
      *handle, transA, transB, m, n, k,
      dpct::get_value(reinterpret_cast<const sycl::double2 *>(alpha), *handle),
      (std::complex<double> *)reinterpret_cast<const sycl::double2 *>(A), ldA,
      (std::complex<double> *)reinterpret_cast<const sycl::double2 *>(B), ldB,
      dpct::get_value(reinterpret_cast<const sycl::double2 *>(beta), *handle),
      (std::complex<double> *)reinterpret_cast<sycl::double2 *>(C), ldC));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// hgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const Half *alpha, const Half *A, int ldA, int loA,
                      const Half *B, int ldB, int loB, const Half *beta,
                      Half *C, int ldC, int loC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasHgemmStridedBatched");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
      *handle, transA, transB, m, n, k,
      dpct::get_value(reinterpret_cast<const sycl::half *>(alpha), *handle),
      reinterpret_cast<const sycl::half *>(A), ldA, loA,
      reinterpret_cast<const sycl::half *>(B), ldB, loB,
      dpct::get_value(reinterpret_cast<const sycl::half *>(beta), *handle),
      reinterpret_cast<sycl::half *>(C), ldC, loC, batch_size));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// sgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const float *alpha, const float *A, int ldA, int loA,
                      const float *B, int ldB, int loB, const float *beta,
                      float *C, int ldC, int loC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasSgemmStridedBatched");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
      *handle, transA, transB, m, n, k, dpct::get_value(alpha, *handle), A, ldA,
      loA, B, ldB, loB, dpct::get_value(beta, *handle), C, ldC, loC,
      batch_size));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// dgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const double *alpha, const double *A, int ldA, int loA,
                      const double *B, int ldB, int loB, const double *beta,
                      double *C, int ldC, int loC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasDgemmStridedBatched");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
      *handle, transA, transB, m, n, k, dpct::get_value(alpha, *handle), A, ldA,
      loA, B, ldB, loB, dpct::get_value(beta, *handle), C, ldC, loC,
      batch_size));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const ComplexFloat *alpha, const ComplexFloat *A, int ldA,
                      int loA, const ComplexFloat *B, int ldB, int loB,
                      const ComplexFloat *beta, ComplexFloat *C, int ldC,
                      int loC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasCgemmStridedBatched");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
      *handle, transA, transB, m, n, k,
      dpct::get_value(reinterpret_cast<const sycl::float2 *>(alpha), *handle),
      (std::complex<float> *)reinterpret_cast<const sycl::float2 *>(A), ldA,
      loA, (std::complex<float> *)reinterpret_cast<const sycl::float2 *>(B),
      ldB, loB,
      dpct::get_value(reinterpret_cast<const sycl::float2 *>(beta), *handle),
      (std::complex<float> *)reinterpret_cast<sycl::float2 *>(C), ldC, loC,
      batch_size));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// zgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const ComplexDouble *alpha, const ComplexDouble *A,
                      int ldA, int loA, const ComplexDouble *B, int ldB,
                      int loB, const ComplexDouble *beta, ComplexDouble *C,
                      int ldC, int loC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasZgemmStridedBatched");

  return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
      *handle, transA, transB, m, n, k,
      dpct::get_value(reinterpret_cast<const sycl::double2 *>(alpha), *handle),
      (std::complex<double> *)reinterpret_cast<const sycl::double2 *>(A), ldA,
      loA, (std::complex<double> *)reinterpret_cast<const sycl::double2 *>(B),
      ldB, loB,
      dpct::get_value(reinterpret_cast<const sycl::double2 *>(beta), *handle),
      (std::complex<double> *)reinterpret_cast<sycl::double2 *>(C), ldC, loC,
      batch_size));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// hgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const Half *alpha, const Half *const A[], int ldA,
                      const Half *const B[], int ldB, const Half *beta,
                      Half *const C[], int ldC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasHgemmBatched");

  return DPCT_CHECK_ERROR(dpct::gemm_batch(
      *handle, transA, transB, m, n, k,
      reinterpret_cast<const sycl::half *>(alpha),
      (const void **)reinterpret_cast<const sycl::half **>(
          const_cast<const Half **>(A)),
      dpct::library_data_t::real_half, ldA,
      (const void **)reinterpret_cast<const sycl::half **>(
          const_cast<const Half **>(B)),
      dpct::library_data_t::real_half, ldB,
      reinterpret_cast<const sycl::half *>(beta),
      (void **)reinterpret_cast<sycl::half **>(const_cast<Half **>(C)),
      dpct::library_data_t::real_half, ldC, batch_size,
      dpct::library_data_t::real_half));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// sgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const float *alpha, const float *const A[], int ldA,
                      const float *const B[], int ldB, const float *beta,
                      float *const C[], int ldC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasSgemmBatched");

  return DPCT_CHECK_ERROR(dpct::gemm_batch(
      *handle, transA, transB, m, n, k, alpha,
      (const void **)const_cast<const float **>(A),
      dpct::library_data_t::real_float, ldA,
      (const void **)const_cast<const float **>(B),
      dpct::library_data_t::real_float, ldB, beta,
      (void **)const_cast<float **>(C), dpct::library_data_t::real_float, ldC,
      batch_size, dpct::library_data_t::real_float));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// dgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const double *alpha, const double *const A[], int ldA,
                      const double *const B[], int ldB, const double *beta,
                      double *const C[], int ldC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasDgemmBatched");

  return DPCT_CHECK_ERROR(dpct::gemm_batch(
      *handle, transA, transB, m, n, k, alpha,
      (const void **)const_cast<const double **>(A),
      dpct::library_data_t::real_double, ldA,
      (const void **)const_cast<const double **>(B),
      dpct::library_data_t::real_double, ldB, beta,
      (void **)const_cast<double **>(C), dpct::library_data_t::real_double, ldC,
      batch_size, dpct::library_data_t::real_double));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const ComplexFloat *alpha, const ComplexFloat *const A[],
                      int ldA, const ComplexFloat *const B[], int ldB,
                      const ComplexFloat *beta, ComplexFloat *const C[],
                      int ldC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasCgemmBatched");

  return DPCT_CHECK_ERROR(
      dpct::gemm_batch(*handle, transA, transB, m, n, k,
                       reinterpret_cast<const sycl::float2 *>(alpha),
                       (const void **)const_cast<const sycl::float2 **>(
                           reinterpret_cast<const sycl::float2 *const *>(A)),
                       dpct::library_data_t::complex_float, ldA,
                       (const void **)const_cast<const sycl::float2 **>(
                           reinterpret_cast<const sycl::float2 *const *>(B)),
                       dpct::library_data_t::complex_float, ldB,
                       reinterpret_cast<const sycl::float2 *>(beta),
                       (void **)const_cast<sycl::float2 **>(
                           reinterpret_cast<sycl::float2 *const *>(C)),
                       dpct::library_data_t::complex_float, ldC, batch_size,
                       dpct::library_data_t::complex_float));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// zgemm
inline int gemm_batch(dpct::queue_ptr handle, oneapi::mkl::transpose transA,
                      oneapi::mkl::transpose transB, int m, int n, int k,
                      const ComplexDouble *alpha,
                      const ComplexDouble *const A[], int ldA,
                      const ComplexDouble *const B[], int ldB,
                      const ComplexDouble *beta, ComplexDouble *const C[],
                      int ldC, int batch_size) try {
  BLAM_DEBUG_OUT("cublasZgemmBatched");

  return DPCT_CHECK_ERROR(
      dpct::gemm_batch(*handle, transA, transB, m, n, k,
                       reinterpret_cast<const sycl::double2 *>(alpha),
                       (const void **)const_cast<const sycl::double2 **>(
                           reinterpret_cast<const sycl::double2 *const *>(A)),
                       dpct::library_data_t::complex_double, ldA,
                       (const void **)const_cast<const sycl::double2 **>(
                           reinterpret_cast<const sycl::double2 *const *>(B)),
                       dpct::library_data_t::complex_double, ldB,
                       reinterpret_cast<const sycl::double2 *>(beta),
                       (void **)const_cast<sycl::double2 **>(
                           reinterpret_cast<sycl::double2 *const *>(C)),
                       dpct::library_data_t::complex_double, ldC, batch_size,
                       dpct::library_data_t::complex_double));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // end namespace cublas
} // end namespace blam

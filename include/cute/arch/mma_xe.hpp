
#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

//fwd declare OCL function and OCL types - should go somewhere else
#include <sycl.hpp> //for sycl::vec

#ifdef __SYCL_DEVICE_ONLY__ 
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
template<class T, int N> using vector_t = typename sycl::vec<T,N>::vector_t;
#else 
#define SYCL_DEVICE_OCL(x) inline x { assert(false); }
template<class T, int N> using vector_t = sycl::vec<T,N>;
#endif

using float8 = vector_t<float, 8>;
using short8 = vector_t<short, 8>;
using ushort8 = vector_t<ushort, 8>;
// using int2 = vector_t<int, 2>; //conflicts with vector_types
using int8 = vector_t<int, 8>;
using uint8 = vector_t<uint, 8>;

SYCL_DEVICE_OCL(float8 intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc));
SYCL_DEVICE_OCL(float  intel_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc));
#undef SYCL_DEVICE_OCL



namespace cute {
//MxNxK_A,B,C,D
//# of vector component of a x subgroup-size x function name
//float8 intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc);
//TODO: Is A really not transposed? Maybe better a macro than separate define for 1,2,4,8
struct XE_8x16x16_BF16BF16F32F32_NN
{
  using DRegisters = float8[1];
  using ARegisters = short8[1];
  using BRegisters = int8[1];
  using CRegisters = float8[1];

  CUTE_HOST_DEVICE static void
  fma(float8      & d,
      short8 const& a,
      int8   const& b,
      float8 const& c)
  {
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
  }
};
//float  intel_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc)
struct XE_1x16x16_BF16BF16F32F32_NN
{
  using DRegisters = float[1];
  using ARegisters = short[1];
  using BRegisters = int8[1];
  using CRegisters = float[1];

  CUTE_HOST_DEVICE static void
  fma(float      & d,
      short const& a,
      int8  const& b,
      float const& c)
  {
    d = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
  }
};
} //namespace cute
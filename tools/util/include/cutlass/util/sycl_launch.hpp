/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Intel All rights reserved.
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

namespace cutlass
{
namespace intel
{
namespace detail
{
template <auto F, typename LaunchPolicy, typename... Args>
void launch(LaunchPolicy launch_policy, sycl::queue q, Args... args) {
  static_assert(syclcompat::args_compatible<LaunchPolicy, F, Args...>,
                "Mismatch between device function signature and supplied "
                "arguments. Have you correctly handled local memory/char*?");
  namespace sycl_exp = sycl::ext::oneapi::experimental;
  sycl_exp::launch_config config(launch_policy.get_range(),
                                 launch_policy.get_launch_properties());

  static_assert(syclcompat::detail::is_nd_range_v<typename LaunchPolicy::RangeT>);
  static_assert(!LaunchPolicy::HasLocalMem);

  auto KF = syclcompat::experimental::detail::KernelFunctor<F, typename LaunchPolicy::RangeT,
                         typename LaunchPolicy::KPropsT,
                         LaunchPolicy::HasLocalMem, Args...>(
        launch_policy.get_kernel_properties(), args...);

  sycl_exp::nd_launch(q, config, KF);
}
}
template <auto F, typename LaunchPolicy, typename... Args>
void launch(LaunchPolicy launch_policy, sycl::queue q, Args... args) {
  static_assert(syclcompat::experimental::detail::is_launch_policy_v<LaunchPolicy>);
  cutlass::intel::detail::launch<F>(launch_policy, q, args...);
}

template <auto F, typename LaunchPolicy, typename... Args>
void launch(LaunchPolicy launch_policy, Args... args) {
  static_assert(syclcompat::experimental::detail::is_launch_policy_v<LaunchPolicy>);
  cutlass::intel::launch<F>(launch_policy, syclcompat::get_default_queue(), args...);
}
}
}

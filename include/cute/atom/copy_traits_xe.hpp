#pragma once

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/arch/copy_xe.hpp>

namespace cute
{
    template<class Tensor>
    struct Copy_Traits<XE_2D_LOAD, Tensor>
    {
        //pass (and store) layout and base
        //compute width_minus_one, height_minus_one,  pitch_minus_one from layout
        //Q: should the src (counting) tensor contain the copy tile layout or should the trait contain it and the src only contain the coord? I think it needs to be in the trait

        Tensor tensor;

        template <class TS, class SLayout,
                  class TD, class DLayout>
        CUTE_HOST_DEVICE friend constexpr void
        copy_unpack(Copy_Traits const &traits,
                    Tensor<TS, SLayout> const &src,
                    Tensor<TD, DLayout> &dst)
        {
            int H = size<0>(tensor);
            int W = size<1>(tensor) * sizeof(tensor::engine_type::value_type);
            XE_2D_LOAD::copy(tensor.data(), W, H, W, src.data().coord_, dst.data());
        }
    };

    template<class Engine, class Layout>
    auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor) {
        using GTensor = Tensor<GEngine, GLayout>;
        using Traits = Copy_Traits<XE_2D_LOAD, GTensor>
        Traits traits{tensor};
        return Copy_Atom<Traits, typename GEngine::value_type>{traits};
    }
}
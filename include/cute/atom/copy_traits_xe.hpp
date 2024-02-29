#pragma once

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/arch/copy_xe.hpp>

namespace cute
{
    template<class GTensor>
    struct Copy_Traits<XE_2D_LOAD, GTensor>
    {
        //pass (and store) layout and base
        //compute width_minus_one, height_minus_one,  pitch_minus_one from layout
        //Q: should the src (counting) tensor contain the copy tile layout or should the trait contain it and the src only contain the coord? I think it needs to be in the trait

        //using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per subggroup) - but static_assert fails
        using ThrID   = Layout<_1>;
        using NumBits = Int<sizeof(typename GTensor::engine_type::value_type)*8>;
        // Map from (src-thr,src-val) to bit
        using SrcLayout = Layout<Shape<_1,NumBits>>;
        // Map from (dst-thr,dst-val) to bit
        using DstLayout = Layout<Shape<_1,NumBits>>;
        // Reference map from (thr,val) to bit
        using RefLayout = SrcLayout;

        GTensor tensor;

        template <class TS, class SLayout,
                  class TD, class DLayout>
        CUTE_HOST_DEVICE friend constexpr void
        copy_unpack(Copy_Traits const &traits,
                    Tensor<TS, SLayout> const &src,
                    Tensor<TD, DLayout> &dst)
        {
            int H = size<0>(traits.tensor);
            int W = size<1>(traits.tensor) * sizeof(typename decltype(traits.tensor)::engine_type::value_type);
            XE_2D_LOAD::copy(traits.tensor.data().get(), W, H, W, src.data().coord_, dst.data().get());
        }
    };

    template<class GEngine, class GLayout>
    auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor) {
        using GTensor = Tensor<GEngine, GLayout>;
        using Traits = Copy_Traits<XE_2D_LOAD, GTensor>;
        Traits traits{gtensor};
        return Copy_Atom<Traits, typename GEngine::value_type>{traits};
    }
}
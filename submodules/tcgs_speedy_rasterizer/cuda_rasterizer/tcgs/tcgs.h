// Copyright (c) 2025 TCGS GROUP. MIT License. See LICENSE for details.
#ifndef TCGS_H
#define TCGS_H

#ifdef _WIN32
    #define TCGS_API __declspec(dllexport)
#else
    #define TCGS_API __attribute__((visibility("default")))
#endif

namespace TCGS
{
    void renderCUDA_Forward(
        const dim3 grid, 
        const dim3 block,
        const uint2* ranges,
        const uint* point_list,
        int width,
        int height,
        int P,
        const float2* means2D,
        const float* features,
        float4* conic_opacity,
        float* final_T,
        uint* n_contrib,
        const float* bg_color,
        float* out_color,
        float* depths,
        float* depth
    );
}

#endif 
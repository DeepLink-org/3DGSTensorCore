// Copyright (c) 2025 TCGS GROUP. MIT License. See LICENSE for details.
#include "tcgs.h"
#include "tcgs_utils.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include "cuda_runtime.h"


namespace cg = cooperative_groups;


using namespace TCGS_UTIL;

__device__ void identify_pixel_properties(
    cg::thread_block &block, int thread_id, int warp_id,
    int width, int height,
    bool &inside, int &tile_id, int &pix_id,
    float2 &pixf_mid, float2 &pixf_local
)
{
    //identify tile ranges
    uint horizontal_blocks = (width + BLOCK_X_TCGS - 1) / BLOCK_X_TCGS;
    uint2 pix_min = make_uint2(block.group_index().x * BLOCK_X_TCGS, block.group_index().y * BLOCK_Y_TCGS);
    uint2 pix_max = make_uint2(min(pix_min.x + BLOCK_X_TCGS, width), min(pix_min.y + BLOCK_Y_TCGS , height));

    //identify global coordinates, each warp contains 8x4 pixels
    uint2 pix = make_uint2(
        pix_min.x + (((warp_id >> 2) << 3) | (thread_id & 7)), 
        pix_min.y + ((thread_id & 127) >> 3));
    
    //identify whether the pixel is inside the image
    inside = pix.x < width && pix.y < height;

    //identify pixel and tile indices
    tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    pix_id = width * pix.y + pix.x;

    //identify local coordinates
    pixf_mid = make_float2(
        __uint2float_rn(pix_min.x) + 7.5f, 
        __uint2float_rn(pix_min.y) + 7.5f
    );
    pixf_local = make_float2(
        __uint2float_rn(pix.x) - pixf_mid.x,
        __uint2float_rn(pix.y) - pixf_mid.y
    );
}


//convert pixel at local coordinates into pixel vectors
__forceinline__ __device__ uint4 pix2vec(
    float2 pixf_local
)
{
    uint4 pixel_vector;
    pixel_vector.x = float22reg(pixf_local.x, pixf_local.y);
    pixel_vector.y = float22reg(pixf_local.x * pixf_local.x, pixf_local.y * pixf_local.y);
    pixel_vector.z = float22reg(1.0f / 3.0f, pixf_local.x * pixf_local.y);
    pixel_vector.w = float22reg(1.0f / 3.0f, 1.0f / 3.0f);
    return pixel_vector;
}

//convert Gaussian primitives into Gaussian vectors
__forceinline__ __device__ uint4 gs2vec(
    float4 conic_opacity,
    float2 means,
    float2 pixf_mid
)
{
    //Converse global coordinates into local coordinates
    means.x = pixf_mid.x - means.x;
    means.y = pixf_mid.y - means.y;

    //compute the constant term of Gaussian vector
    float constant = 
        conic_opacity.w + 
        conic_opacity.x * means.x * means.x + 
        conic_opacity.y * means.x * means.y +
        conic_opacity.z * means.y * means.y;
    
    //compute the gaussian vector
    uint4 gaussian_vector;
    gaussian_vector.x = float22reg(
        2.0f * conic_opacity.x * means.x + conic_opacity.y * means.y,
        2.0f * conic_opacity.z * means.y + conic_opacity.y * means.x
    );
    gaussian_vector.y = float22reg(conic_opacity.x, conic_opacity.z);
    gaussian_vector.z = float22reg(constant, conic_opacity.y);
    gaussian_vector.w = float22reg(constant, constant);
    return gaussian_vector;
}

//Stack vectors into matrices
__forceinline__ __device__ void vec2mat(
    uint4 vecs,
    uint* mat_smem,
    uint thread_id
)
{
    uint addr = (thread_id << 2);
    mat_smem[addr] = vecs.x;
    mat_smem[addr | 1] = vecs.y;
    mat_smem[addr | 2] = vecs.z;
    mat_smem[addr | 3] = vecs.w;
}

//Store exponents registers into matrix
__forceinline__ __device__ void store_exponent_mat(
    uint reg0, uint reg1, uint reg2, uint reg3,
    uint* mat_smem
)
{
    mat_smem[0] = reg0;
    mat_smem[8] = reg1;
    mat_smem[16] = reg2;
    mat_smem[24] = reg3;
}

//Culling and Alpha-blending
__forceinline__ __device__ uint2 culling_and_blending(
    uint* exponent_matrix, // Shared Memory
    uint2* channels_smem,
    half &T, int gs_index, int thread_id, uint2 RGBD
)
{
#pragma unroll
    for(int k = 0; k < 8; ++k)
    {
        half2 exponents = *reinterpret_cast<half2*>(exponent_matrix + ((k<<8)|thread_id));
        //beta > -ln(225) and beta < 0, otherwise culled
        if(__hgt(exponents.x, __float2half_rn(-7.995f)) && __hlt(exponents.x, __float2half_rn(0.0f)))
        {
            //alpha-blending
            half alpha = __hmul(T, __hmin(__float2half_rn(0.99f), fast_ex2_f16(exponents.x)));
            T = __hsub(T, alpha);
            uint2 channel = channels_smem[gs_index|(k<<1)];
            uint alpha2 = half22uint(make_half2(alpha, alpha));
            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }

        if(__hgt(exponents.y, __float2half_rn(-7.995f)) && __hlt(exponents.y, __float2half_rn(0.0f)))
        {
            //alpha-blending
            half alpha = __hmul(T, __hmin(__float2half_rn(0.99f), fast_ex2_f16(exponents.y)));
            T = __hsub(T, alpha);
            uint2 channel = channels_smem[gs_index|(k<<1)|1];
            uint alpha2 = half22uint(make_half2(alpha, alpha));
            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }
    }
    return RGBD;
}

//Main Kernel of TC-GS
__global__ void renderCUDA_TCGS(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width, int height,
    const float2* __restrict__ points_xy_image,
    const uint2* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ invdepth
) 
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank();
    int warp_id = thread_id / WAPR_SIZE;

    bool inside; //Whether the pixel is insice the image
    int pix_id;  //Pixel index of the screen space
    int tile_id; //Tile index of the screen space
    float2 pixf_mid; //Center pixel inside the tile
    float2 pixf_local; //Local coordinates of the pixel

    identify_pixel_properties(
        block, thread_id, warp_id,
        width, height,
        inside, tile_id, pix_id, pixf_mid, pixf_local
    );

    //Define Matrices
    __shared__ uint multiuse_matrix[BLOCK_SIZE_TCGS * REDUCE_SIZE / 2]; //multiple use for storing 
    __shared__ uint exponent_matrix[BLOCK_SIZE_TCGS * VECTOR_SIZE];
    __shared__ uint2 channels_smem[BLOCK_SIZE_TCGS];
    uint* exponent_matrix_addr = exponent_matrix + ((thread_id&3) << 8) + (warp_id<<5) + ((thread_id&31) >> 2);
    uint pixmat_reg[4];

    //load pixel_vector
    uint4 pix_vec = pix2vec(pixf_local);
    vec2mat(pix_vec, multiuse_matrix, thread_id);
    __syncwarp();
    load_matrix_x4(
        pixmat_reg[0], pixmat_reg[1], pixmat_reg[2], pixmat_reg[3],
        multiuse_matrix + (thread_id<<2)
    );

    bool done = !inside;
    bool warp_done = (__ballot_sync(~0, done) == (~0));
    uint2 range = ranges[tile_id];
    int toDo = range.y - range.x;
    const int rounds = (toDo + BLOCK_SIZE_TCGS - 1) / BLOCK_SIZE_TCGS;

    half T = __float2half_rn(1.0f);
    uint num_contrib = 0u;
    uint2 RGBD = make_uint2(0u, 0u);

    for(int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE_TCGS)
    {
        int num_done = __syncthreads_count(done);
        if(num_done == BLOCK_SIZE_TCGS)
            break;

        int progress = ((i<<8) | thread_id);
        uint4 gaussian_vec = make_uint4(0u, 0u, 0u, float22reg(1.0, 1.0));
        uint2 channel = make_uint2(0u, 0u);
        
        //load channels ans gaussian vectors
        if(range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            float4 conics = conic_opacity[coll_id];
            float2 means = points_xy_image[coll_id];
            gaussian_vec = gs2vec(conics, means, pixf_mid);
            channel = features[coll_id];
        }
        vec2mat(gaussian_vec, multiuse_matrix, thread_id);
        channels_smem[thread_id] = channel;

        block.sync();
        const int gs_num = min(BLOCK_SIZE_TCGS, toDo);
        for(int j = 0; !warp_done && j < gs_num; j += REDUCE_SIZE){
            //1. load the Gaussian matrix V
            uint gsmat_reg[2];
            load_matrix_x2(gsmat_reg[0], gsmat_reg[1], multiuse_matrix + (j<<2) + ((thread_id&31)<<2));

            //2. compute exponents matrix B = U^T * V = [U1, U2]^T * [V1, V2]
            //The division strategy is to align the format of instructions in ptx
            //2.1 Compute B_1 = [U1^T * V1, U2^T * V1]
            uint expmat_reg[4] = {0u, 0u, 0u, 0u};
            mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
                pixmat_reg[0], pixmat_reg[1], gsmat_reg[0], expmat_reg[0], expmat_reg[1]);
            mma_16x8x8_f16_f16(expmat_reg[2], expmat_reg[3],
                pixmat_reg[2], pixmat_reg[3], gsmat_reg[0], expmat_reg[2], expmat_reg[3]);
            store_exponent_mat(expmat_reg[0], expmat_reg[1], expmat_reg[2], expmat_reg[3], exponent_matrix_addr);
            __syncwarp();
            
            //2.2 Compute B_2 = [U1^T * V2, U2^T * V2]
            expmat_reg[0] = expmat_reg[1] = expmat_reg[2] = expmat_reg[3] = 0u;
            mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
                pixmat_reg[0], pixmat_reg[1], gsmat_reg[1], expmat_reg[0], expmat_reg[1]);
            mma_16x8x8_f16_f16(expmat_reg[2], expmat_reg[3],
                pixmat_reg[2], pixmat_reg[3], gsmat_reg[1], expmat_reg[2], expmat_reg[3]);
            store_exponent_mat(expmat_reg[0], expmat_reg[1], expmat_reg[2], expmat_reg[3], exponent_matrix_addr + 1024);
            __syncwarp();

            //3. culling and alpha-blending
            RGBD = culling_and_blending(exponent_matrix, channels_smem, T, j, thread_id, RGBD);
            if(__hlt(T, __float2half_rn(0.0001f)))
                done = true;
            if(__ballot_sync(~0, done) == (~0))
                warp_done = true;
            num_contrib += REDUCE_SIZE;
        }
    }

    //output colors and other informations
    if(inside)
    {
        float Tf = __half2float(T);
        half2 RGh = uint2half2(RGBD.x);
        half2 BDh = uint2half2(RGBD.y);
        final_T[pix_id] = Tf;

        out_color[pix_id] = __half2float(RGh.x) + Tf * bg_color[0];
        out_color[pix_id + width * height] = __half2float(RGh.y) + Tf * bg_color[1];
        out_color[pix_id + 2 * width * height] = __half2float(BDh.x) + Tf * bg_color[2];
        if(invdepth)
            invdepth[pix_id] = __half2float(BDh.y);
        
        n_contrib[pix_id] = num_contrib;
    }
}


__global__ void transform_coefs(
    const int P,
    const float* colors,
    const float* depths,
    float4* conic_opacity,
    uint2* feature_encoded,
    float* invdepth
)
{
    auto idx = cg::this_grid().thread_rank();
    if(idx >= P)
        return;
    
    //Preprocess the inverse covariance for vectorizing 
    float4 conics = conic_opacity[idx];
    conic_opacity[idx] = make_float4(
        LOG2E_N_2 * conics.x, LOG2E_N * conics.y, LOG2E_N_2 * conics.z, fast_lg2_f32(conics.w)
    );

    //Compress the colors into fp16
    float4 features = make_float4(
        colors[idx * 3], colors[idx * 3 + 1], colors[idx * 3 + 2], 0.0f
    );
    if(invdepth)
        features.w = 1.0f / depths[idx];
    uint RG = float22reg(features.x, features.y);
    uint BD = float22reg(features.z, features.w);
    feature_encoded[idx] = make_uint2(RG, BD);
}


void TCGS::renderCUDA_Forward(
    const dim3 grid,
    const dim3 block,
    const uint2* ranges,
    const uint* point_list,
    int width,
    int height,
    int P,
    const float2* means2D,
    const float* colors,
    float4* conic_opacity,
    float* final_T,
    uint* n_contrib,
    const float* bg_color,
    float* out_color,
    float* depths,
    float* depth
)
{
    //Preprocess for TCGS
    uint2* feature_encoded = nullptr;
    cudaMalloc(&feature_encoded, P * sizeof(uint2));
    transform_coefs<< <(P + 255) / 256, 256>> >(
        P, colors, depths, conic_opacity, feature_encoded, depth
    );

    //Running TCGS
    renderCUDA_TCGS<< <grid, block>> >(
        ranges, point_list,
        width, height,
        means2D, feature_encoded, conic_opacity,
        final_T, n_contrib,
        bg_color, out_color, depth
    );
    //
    cudaFree(feature_encoded);
    
}

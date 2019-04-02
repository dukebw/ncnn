// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "bilateralslice.h"
#include <assert.h>

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

namespace ncnn {

DEFINE_LAYER_CREATOR(BilateralSlice)

struct GridSizes {
        int64_t h;
        int64_t w;
        int64_t bs;
        int64_t coeffs_chans;
        int64_t gd;
        int64_t gh;
        int64_t gw;
        int64_t input_chans;
};

BilateralSlice::BilateralSlice()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
}

int BilateralSlice::load_param(const ParamDict& pd)
{
    has_offset = pd.get(0, 0);
    num_luma_bins = pd.get(1, 0);

    return 0;
}

static float
diff_abs(float x)
{
        float eps = 1e-8;
        return sqrt(x*x+eps);
}

static float
weight_z(float x)
{
        float abx = diff_abs(x);
        return max(1.0f - abx, 0.0f);
}

template <typename T>
static void
bilateral_slice_cpu_forward_kernel(T * __restrict__ output,
                                   T * __restrict__ bilateral_grid,
                                   T * __restrict__ guide,
                                   T * __restrict__ input,
                                   GridSizes gsz,
                                   bool has_offset,
                                   int total_count,
                                   int output_chans)
{
        int h = gsz.h;
        int w = gsz.w;
        int gd = gsz.gd;
        int gh = gsz.gh;
        int gw = gsz.gw;
        int input_chans = gsz.input_chans;
        int coeff_stride = input_chans;
        int grid_chans = input_chans*output_chans;

        if (has_offset) {
                grid_chans += output_chans;
                coeff_stride += 1;
        }

        for (int idx = 0;
             idx < total_count;
             ++idx) {
                int x = idx % w;
                int y = (idx / w) % h;
                int out_c = (idx / (h*w)) % output_chans;
                int b = (idx / (output_chans*w*h));

                float gx = (x + 0.5f)*gw/(1.0f*w);
                float gy = (y + 0.5f)*gh/(1.0f*h);
                float gz = guide[x + w*(y + h*b)]*gd;

                int fx = static_cast<int>(floor(gx - 0.5f));
                int fy = static_cast<int>(floor(gy - 0.5f));
                int fz = static_cast<int>(floor(gz - 0.5f));


                // Grid strides
                int sy = gw;
                int sz = gw*gh;
                int sc = gd*gw*gh;
                int sb = grid_chans*gd*gw*gh;

                float value = 0.0f;
                for (int in_c = 0;
                     in_c < coeff_stride;
                     ++in_c) {
                        float coeff_sample = 0.0f;

                        for (int xx = fx; xx < fx + 2; ++xx) {
                                int x_ = max(min(xx, gw - 1), 0);
                                float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

                                for (int yy = fy; yy < fy + 2; ++yy) {
                                        int y_ = max(min(yy, gh - 1), 0);
                                        float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

                                        for (int zz = fz; zz < fz + 2; ++zz) {
                                                int z_ = max(min(zz, gd - 1), 0);
                                                float wz = weight_z(zz + 0.5 - gz);
                                                int c_ = coeff_stride*out_c + in_c;
                                                int grid_idx = x_ + sy*y_ + sz*z_ + sc*c_ + sb*b;

                                                coeff_sample += bilateral_grid[grid_idx]*wx*wy*wz;
                                        }
                                }
                        } // Grid trilinear interpolation
                        if (in_c < input_chans) {
                                int input_idx = x + w*(y + h*(in_c + input_chans*b));
                                value += coeff_sample*input[input_idx];
                        } else { // Offset term
                                value += coeff_sample;
                        }
                }

                output[idx] = value;
        }

}

int
BilateralSlice::forward(const std::vector<Mat>& bottom_blobs,
                        std::vector<Mat>& top_blobs,
                        const Option& opt) const
{
        const Mat& bilateral_grid = bottom_blobs[0];
        const Mat& guide = bottom_blobs[1];
        const Mat& input = bottom_blobs[2];
        fprintf(stderr, "BilateralSlice\n");

        /**
         * NOTE(brendan): CHW where C = num_luma_bins * (n_in + 1) * n_out,
         * where n_in = 3 and n_out = 3 for RGB images.
         */
        assert(bilateral_grid.dims == 3);
        assert(guide.dims == 2);
        assert(input.dims == 3);
        constexpr int64_t coeffs_chans = 3*4;
        assert((input.w == guide.w) && (input.h == guide.h));
        assert(bilateral_grid.c == coeffs_chans*num_luma_bins);

        constexpr int64_t bs = 1;
        int64_t h = guide.h;
        int64_t w = guide.w;
        int64_t gd = num_luma_bins;
        int64_t gh = bilateral_grid.h;
        int64_t gw = bilateral_grid.w;
        int64_t input_chans = input.c;
        GridSizes grid_sizes{.h = h,
                             .w = w,
                             .bs = bs,
                             .coeffs_chans = coeffs_chans,
                             .gd = gd,
                             .gh = gh,
                             .gw = gw,
                             .input_chans = input_chans};

        int64_t output_chans;
        if (has_offset) {
                assert((coeffs_chans % (input_chans + 1)) == 0);
                output_chans = coeffs_chans/(input_chans + 1);
        } else {
                assert((coeffs_chans % input_chans) == 0);
                output_chans = coeffs_chans / input_chans;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, output_chans, input.elemsize, opt.blob_allocator);

        bilateral_slice_cpu_forward_kernel((float *)top_blob.data,
                                           (float *)bilateral_grid.data,
                                           (float *)guide.data,
                                           (float *)input.data,
                                           grid_sizes,
                                           has_offset,
                                           bs*output_chans*h*w,
                                           output_chans);

        return 0;
}

} // namespace ncnn

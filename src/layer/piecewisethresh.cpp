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

#include "piecewisethresh.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(PiecewiseThresh)

PiecewiseThresh::PiecewiseThresh()
{
}

int PiecewiseThresh::load_param(const ParamDict& pd)
{
        return 0;
}

int PiecewiseThresh::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;
        constexpr float threshs[] = {0.0f, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
                                     0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375};
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0;
             q < channels;
             ++q) {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0;
                     i < size;
                     ++i) {
                        float accum = 0.0f;
                        for (uint32_t thresh_i = 0;
                             thresh_i < sizeof(threshs)/sizeof(threshs[0]);
                             ++thresh_i) {
                                if (ptr[i] > threshs[thresh_i])
                                        /* TODO(brendan): learned slopes */
                                        accum += ptr[i];
                        }
                        ptr[i] = accum;
                }
        }

        return 0;
}

} // namespace ncnn

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
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = false;
}

// int PiecewiseThresh::load_param(const ParamDict& pd)
// {
//         return 0;
// }

int PiecewiseThresh::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;
        // constexpr float slopes[] = {1.0330992900, 0.0342797115,	0.0342896283, 0.0302721635, 0.0272607822,
        // 0.0198369585,0.0124553759,0.0061479830, -0.0006521735,	-0.0067093680,	-0.0124623375,	
        // -0.0187911615, -0.0271771401,-0.0426548049,-0.0735674500,-0.2092483190,
        // 0.9849609730,-0.0199312363,-0.0196038391,-0.0160276759,-0.0143460752,	
        // -0.0103229443,	-0.0104410145,	-0.0115587469, -0.0123622930,-0.0143735185,	
        // -0.0173819494,	-0.0213875696,-0.0243862700,-0.0443304256,-0.1241403890,	
        // -0.2320523710, 0.9678404330,-0.0396916009,-0.0404757559,-0.0398040749,-0.0392591506,	
        // -0.0356815979,	-0.0303876083,	-0.0260948539,-0.0205931533,	
        // -0.0206054375,	-0.0331127532,	-0.0257681720,0.0256281700,	
        // -0.0524809472,	-0.0795733556,	-0.0747298673};

//      constexpr float threshs[] = {-0.02515548,  0.02662835,  0.05552383,  0.14699794,  0.15546258,
//      0.16602647,  0.25545067,  0.38749704,  0.49660265,  0.5762779,
//      0.63344246,  0.67650753,  0.7177053,   0.74794936,  0.74488586,
//      0.6955773,   -0.02414442,  0.04989728,  0.06737315,  0.10356149,  
//      0.15711074,  0.29212543,  0.3598683,   0.41364187,  0.4739961,  
//      0.5471802,   0.59571785,  0.5896928,   0.70380616,  0.7527259,   
//      0.7147031,   0.8440116,   -0.02377157,  0.0413614,   0.06601016,  
//      0.0846173,   0.10090853,  0.1294813,   0.17295782,  0.22764263,  
//      0.34187758,  0.40172493,  0.2724438,   0.48649594,  0.6027179,   
//      0.56251246,  0.6227107,   0.90343297};

        constexpr float slopes[3][16] = {{1.0330992900, 0.0342797115,	0.0342896283, 0.0302721635, 0.0272607822,
        0.0198369585,0.0124553759,0.0061479830, -0.0006521735,	-0.0067093680,	-0.0124623375,	
        -0.0187911615, -0.0271771401,-0.0426548049,-0.0735674500,-0.2092483190}, {0.9849609730,-0.0199312363,-0.0196038391,-0.0160276759,-0.0143460752,	
         -0.0103229443,	-0.0104410145,	-0.0115587469, -0.0123622930,-0.0143735185,	
         -0.0173819494,	-0.0213875696,-0.0243862700,-0.0443304256,-0.1241403890,	
         -0.2320523710},{0.9678404330,-0.0396916009,-0.0404757559,-0.0398040749,-0.0392591506,	
         -0.0356815979,	-0.0303876083,	-0.0260948539,-0.0205931533,	
         -0.0206054375,	-0.0331127532,	-0.0257681720,-0.0256281700,	
         -0.0524809472,	-0.0795733556,	-0.0747298673}};

             constexpr float threshs[3][16] = {{-0.02515548,  0.02662835,  0.05552383,  0.14699794,  0.15546258,
     0.16602647,  0.25545067,  0.38749704,  0.49660265,  0.5762779,
     0.63344246,  0.67650753,  0.7177053,   0.74794936,  0.74488586,
     0.6955773},{-0.02414442,  0.04989728,  0.06737315,  0.10356149,  
      0.15711074,  0.29212543,  0.3598683,   0.41364187,  0.4739961,  
      0.5471802,   0.59571785,  0.5896928,   0.70380616,  0.7527259,   
      0.7147031,   0.8440116},{-0.02377157,  0.0413614,   0.06601016,  
      0.0846173,   0.10090853,  0.1294813,   0.17295782,  0.22764263,  
      0.34187758,  0.40172493,  0.2724438,   0.48649594,  0.6027179,   
      0.56251246,  0.6227107,   0.90343297}};
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
                             thresh_i < 16;
                             ++thresh_i) {
                                if (ptr[i] > threshs[q][thresh_i])
                                        /* TODO(brendan): learned slopes */
                                        accum += ((ptr[i] - threshs[q][thresh_i]) * slopes[q][thresh_i]);
                        }
                        ptr[i] = accum;
                }
        }

        return 0;
}

} // namespace ncnn

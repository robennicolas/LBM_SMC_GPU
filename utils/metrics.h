#pragma once
#include <iostream>
#include "../include/lbm/lbm_gpu.h"

namespace Metrics {

    class metrics {

    private:
        int w_, h_, warmup_steps_;
        int probe_idx_;
        float u0_;                      
        Physics::lbm& sim_;            


        // Strouhal tracking
        float uy_prev_;
        int last_crossing_;
        int crossing_count_;
        float freq_sum_;

    public:
        metrics(int w, int h, int warmup_steps, int xtop, float u0, Physics::lbm& sim);
        
        void strouhalNumCompute(int t);
        
        float MLUPS(int iterations, double elapsed_ms);

    };


}
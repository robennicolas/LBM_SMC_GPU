#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h> 
#include "../lbm/lbm_gpu.h"


namespace smc{

    class sir{

        private:
        int N_;  //samples size
        int Nx_, Nz_;  // state and measurement size
        float a_;
        float u0_;
        float min_dist_ = 0.0f;
        


        //----DEVICE----
        float* d_z_; //measurement
        float* d_x_sample_; //sample vectors
        float* d_z_sample_; //sample vectors
        float* d_w_; //weights
        float* d_dist_; //dist device
        curandState* d_states_;// random number for normal distribution from cuda


        //----HOST----
        float* h_x_; // position
        float* h_dist_; //dist host
        float* h_x_sample_; //sample vectors
        float* h_w_; //weights


        Physics::lbm& myFluid_; //


        public:
        sir(Physics::lbm& myFluid, int N, int Nx, int Nz, float a , float u0, std::mt19937& gen);
        ~sir();

        void run_smc(std::mt19937& gen);
        void normalization();
        float computeESS();
        void estimation();
        void resampling(std::mt19937& gen);
        float* getState(){ return h_x_;}
    };

}


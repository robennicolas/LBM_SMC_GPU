#pragma once 
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>


namespace Physics{

    class lbm{
        
        private:    

            // Simulation constants

            int d_;               //dimension
            int q_;               //d2q9 method with 9 direction
            int height_;           //grid height
            int width_;           //grid width    
            int N_;
            float viscosity_;     //viscosity
            float omega_;         //relaxation parameter (a function of viscosity)
            float u0_;            //initial velocity

            float obst_pos_[2];
            

            float* d_f_;    //old lattice | vec dim 9 * height_ * width_
            float* d_f_next_;     //new lattice | vec dim 9 * height_ * width_
            float* d_ux_;       //x velocity lattice | vec dim height_ * width_
            float* d_uy_;       //y velocity alttice | vec dim height_ * width_
            float* d_speed_;    //cell squared velocity lattice | vec dim height_ * width_
            float* d_rho_;     //density  | vec height_ * width_
            float* d_bar_;     //barrier  | vec height_ * width_

            float* ux_;       //x velocity lattice | vec dim height_ * width_
            float* uy_;       //y velocity alttice | vec dim height_ * width_
            float* speed_;    //cell squared velocity lattice | vec dim height_ * width_
            float* bar_;     //barrier  | vec height_ * width_

            size_t grid_bytes;
            size_t f_bytes;
            
        public:

            lbm(int d, int q, int height, int width, float viscosity, float u0);
            ~lbm();

            void runInit(int xtop, int ytop, int yheight);

            void runPullStreaming();

            void runCollision();

            float* getDeviceSpeed() const { return d_speed_; }
            float* getDeviceBar()   const { return d_bar_;   }
            float* getDevice_ux_()  const { return d_ux_;    }
            float* getDevice_uy_()  const { return d_uy_;    }


            
            const std::vector<int> getGridDim() const { return {width_, height_}; } //getter
            const float* getObstaclePos() const { return obst_pos_; }



    };
}

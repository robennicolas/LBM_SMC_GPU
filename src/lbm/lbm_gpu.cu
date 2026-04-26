#include "../../include/lbm/lbm_gpu.h"

// constant déclariation in the cu file for the CUDA compiler
__constant__ int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};  //opposite table to find the oposite of a direction in d2q9
__constant__ int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ float weights_[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};        // a constant talbe fro the weight d2q9


//index function
__device__ int IX(int f, int x, int y, int w, int h){return f * (w * h) + (x + y * w) ;}  
                                                                                            


                                                                                            
__global__ void init_kernel(float* f, float* f_next, float* bar,
                                     float omega, float u0, 
                                     int xtop, int ytop, int yheight, 
                                     int w, int h){

    float u2 = u0*u0;
    float rho_init = 1.0f;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h){

        int cell_idx = x + y * w;

        for (int k = 0; k < 9; k++) {

            float dot = cx[k] * u0;     //dot product with the direction in x for our fluid to move from west to east
            f[IX(k, x, y, w , h)] = weights_[k] * rho_init * (1.0f + 3.0f*dot + 4.5f*dot*dot - 1.5f*u2);
            f_next[IX(k, x, y, w , h)] = f[IX(k, x, y, w , h)];

                    
            
        }
        if (x >= xtop && x < xtop + 5 && y >= ytop && y < ytop + yheight){
            bar[cell_idx] = 1.0f;
        } else {
            bar[cell_idx] = 0.0f;
        }
    }         
}




__global__ void pullStreaming_kernel(float* f, float* f_next, float* bar, float u0,
                                     int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int i = x + y * w;
    if (bar[i]) return;                     // obstacle: do nothing (bounce‑back handled by neighbors)

    // pull from neighbors (with periodic Y, no wrap in X)
    f_next[IX(0, x, y, w, h)] = f[IX(0, x, y, w, h)];   // rest direction

    for (int k = 1; k < 9; k++) {
        int prev_x = x - cx[k];
        int prev_y = (y - cy[k] + h) % h;    // periodic in Y only

        // Out of bounds in X -> handled later by inlet/outlet BC
        if (prev_x < 0 || prev_x >= w) continue;

        int prev_idx = prev_x + prev_y * w;
        if (bar[prev_idx]) {
            // Bounce‑back: reflect from current cell's opposite direction
            f_next[IX(k, x, y, w, h)] = f[IX(opposite[k], x, y, w, h)];
        } else {
            f_next[IX(k, x, y, w, h)] = f[IX(k, prev_x, prev_y, w, h)];
        }
    }

    // inlet (x == 0) – equilibrium with velocity u0
    if (x == 0) {
        float u2 = u0 * u0;
        for (int k = 0; k < 9; k++) {
            float dot = cx[k] * u0;
            f_next[IX(k, 0, y, w, h)] = weights_[k] * (1.0f + 3.0f*dot + 4.5f*dot*dot - 1.5f*u2);
        }
    }

    // outlet (x == w-1) – zero gradient: copy from interior
    if (x == w-1) {
        for (int k = 0; k < 9; k++) {
            f_next[IX(k, w-1, y, w, h)] = f[IX(k, w-2, y, w, h)];
        }
    }
}



__global__ void collision_kernel(float* f, float* ux, float* uy,
                                float* rho, float* bar, float* speed,
                                int w, int h, float omega){

                                
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h){
        int i = x + y * w;
        if (bar[i] == 0) {//with we're not on the barrier
            // initialize to zero
            rho[i] = 0.0f;
            ux[i]  = 0.0f;
            uy[i]  = 0.0f;

            // rho compute on one pixel of the f
            for(int k = 0; k < 9; k++){
                rho[i] += f[IX(k,x,y,w,h)];
            }
                    
            if(rho[i] > 0){
            //only if rho !=0
                for(int k = 0; k < 9; k++){ //speed compute on one pixel
                    ux[i] += f[IX(k,x,y,w,h)] * cx[k];
                    uy[i] += f[IX(k,x,y,w,h)] * cy[k];
                }
                ux[i] /= rho[i];//compute after for efficiency
                uy[i] /= rho[i];

                float u2 = ux[i]*ux[i] + uy[i]*uy[i]; //some computation before loop for efficiency
                speed[i] = sqrt(u2);
                        
                // 4. Update BGK 
                for(int k = 0; k < 9; k++){
                    float dot = ux[i]*cx[k] + uy[i]*cy[k];
                    float feq = weights_[k] * rho[i] * (1.0f + 3.0f*dot + 4.5f*dot*dot - 1.5f*u2);
                    f[IX(k,x,y,w,h)] += omega * (feq - f[IX(k,x,y,w,h)]);
                }
            }


        } else {
            speed[i] = 0.0f; // for displaying a black wall in the renderer
        }  


    }
}

namespace Physics {

    lbm::lbm(int d, int q, int height, int width, float viscosity, float u0) 
    : d_(d), q_(q), height_(height), width_(width), N_(height * width), viscosity_(viscosity), omega_(1.0f/(3.0f *viscosity + 0.5f)), u0_(u0), 
    grid_bytes(width_ * height_ * sizeof(float)) , f_bytes(9 * grid_bytes)
    {

        ux_    = (float*)malloc(grid_bytes);
        uy_    = (float*)malloc(grid_bytes);
        speed_ = (float*)malloc(grid_bytes);
        bar_   = (float*)malloc(grid_bytes);

        cudaMalloc(&d_f_, f_bytes);
        cudaMalloc(&d_f_next_, f_bytes);
        cudaMalloc(&d_ux_, grid_bytes);
        cudaMalloc(&d_uy_,grid_bytes);
        cudaMalloc(&d_speed_,grid_bytes);
        cudaMalloc(&d_rho_,grid_bytes);
        cudaMalloc(&d_bar_,grid_bytes);

    }





    lbm::~lbm() {
        free(ux_);
        free(uy_);
        free(speed_);
        free(bar_);

        cudaFree(d_f_);
        cudaFree(d_f_next_);
        cudaFree(d_ux_);
        cudaFree(d_uy_);
        cudaFree(d_speed_);
        cudaFree(d_rho_);
        cudaFree(d_bar_);
    }

    


    

    void lbm::runInit(int xtop, int ytop, int yheight){
        obst_pos_[0] = xtop + 5.0f; //because the obstacle width is define in the loop as 10, so we choose the middle of the obstacle = 10/2
        obst_pos_[1] = int(ytop + yheight * 0.5f); // same logic so yheight/2

        dim3 threadsPerBlock(16,16) , numBlocks((width_ + 15)/ 16 , (height_ + 15)/ 16);

        init_kernel<<<numBlocks , threadsPerBlock>>> (d_f_, d_f_next_, d_bar_,
                                                            omega_, u0_,
                                                            xtop,  ytop,  yheight, 
                                                            width_, height_);                                
    }





    void lbm::runPullStreaming(){

        dim3 threadsPerBlock(16,16) , numBlocks((width_ + 15)/ 16 , (height_ + 15)/ 16);

        pullStreaming_kernel<<<numBlocks , threadsPerBlock>>> (d_f_, d_f_next_, d_bar_, u0_,
                                                            width_, height_);

        std::swap(d_f_, d_f_next_);                                                                                    
    }

    

    void lbm::runCollision(){

        dim3 threadsPerBlock(16,16) , numBlocks((width_ + 15)/ 16 , (height_ + 15)/ 16);


        collision_kernel<<<numBlocks , threadsPerBlock>>> (d_f_, d_ux_, d_uy_,
                                                           d_rho_, d_bar_, d_speed_,
                                                            width_, height_, omega_);                                                                         
    }



}





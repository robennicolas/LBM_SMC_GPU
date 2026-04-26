#include "../../include/filter/smc_gpu.h"
#include <curand.h>
#include <curand_kernel.h>

__constant__  float sensor_pos[10] = {  //sensor position not to far or to close from the real obstacle 
            115, 80,   
            115, 100,
            115, 120,
            140, 80,
            140, 120,
        };
        
__constant__ float processNoise = 0.5f;
__constant__ float measurementNoise = 0.05f;




__device__ void predictMeasurement(float* d_x_sample,float* d_z_sample, int i, int Nx, int Nz, float a, float u0) {
    int num_sensors = static_cast<int>(Nz / Nx); // Nx is the position dimension, each sensor contain a position x,y in 2d so its dim is Nz/2
    int ix = i * Nx; // Offset

    for (int j = 0; j < num_sensors; j++) { //for simplicity we wont add another loop on Nx but assume the position is 2
        // Offset
        int iz = i * Nz + j * Nx;

        // Relative position
        float dx = sensor_pos[j * Nx] - d_x_sample[ix];
        float dy = sensor_pos[j * Nx + 1] - d_x_sample[ix + 1];  

        float r2 = dx * dx + dy * dy;
        float R2 = (a * a) * 0.25f; // Radius squared

        float r4 = r2 * r2;

        if(r2 < R2 || r2 < 1e-6f) {  //security check
            d_z_sample[iz]     = 0.0f;
            d_z_sample[iz + 1] = 0.0f;
            continue;
        }

        // (Potential Flow)
        d_z_sample[iz]     = u0 * (1.0f - (R2 * (dx * dx - dy * dy)) / r4);
        d_z_sample[iz + 1] = u0 * (-R2 * (2.0f * dx * dy)) / r4;
    }
}



__global__ void setup_kernel(curandState* states, unsigned long seed, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}



__global__ void prediction_kernel(curandState* states, float* d_x_sample, int Nx, int N) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        curandState local = states[i];
        for (int k = 0; k < Nx; k++) {
            float noise = curand_normal(&local) * processNoise;
            d_x_sample[i * Nx + k] += noise;
        }
        states[i] = local;  
    }
}



__global__ void weigthUpdate(int N, int Nx, int Nz, float a, float u0, float min_dist, float* d_z, float* d_x_sample, float* d_z_sample, float* d_dist, float* d_w) {
    // Predict measurement for each particle
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N){ 
        predictMeasurement(d_x_sample, d_z_sample, i, Nx, Nz, a, u0);
    }  

    // Compute raw squared distances (no sigma normalization yet)
    if (i < N){  
        float d = 0.0f;
        for(int k = 0; k < Nz; k++) { 
            float diff = d_z_sample[Nz * i + k] - d_z[k];
            d += diff * diff; 
        }   
        d_dist[i] = d;
    }

    // Compute likelihood with sigma = measurementNoise scaled to distance range
    float inv_2sig2 = 1.0f / (2.0f * measurementNoise * measurementNoise);

    if (i < N){ 
        d_w[i] = expf(-(d_dist[i] - min_dist) * inv_2sig2);
    }
}


__global__ void sample_velocity_kernel(float* d_ux, float* d_uy, float* d_z,
                                        int num_sensors, int w) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < num_sensors) {
        int x = (int)roundf(sensor_pos[j * 2]);
        int y = (int)roundf(sensor_pos[j * 2 + 1]);
        int idx = x + y * w;
        d_z[j * 2]     = d_ux[idx];
        d_z[j * 2 + 1] = d_uy[idx];
    }
}



namespace smc{


    sir::sir(Physics::lbm& myFluid, int N, int Nx, int Nz,float a, float u0, std::mt19937& gen):
    myFluid_(myFluid),
    N_(N), Nx_(Nx), Nz_(Nz) , a_(a), u0_(u0)

    {           
        //----DEVICE----
        cudaMalloc(&d_z_, Nz_ * sizeof(float));   
        cudaMalloc(&d_z_sample_, N_ * Nz_ * sizeof(float));   
        cudaMalloc(&d_x_sample_, N_ * Nx_ * sizeof(float));   
        cudaMalloc(&d_w_, N_ * sizeof(float));   
        cudaMalloc(&d_dist_, N_ * sizeof(float));   

        cudaMalloc(&d_states_, N_ * sizeof(curandState));
        setup_kernel<<<(N_+15)/16, 16>>>(d_states_, time(NULL), N_);


        //----HOST----
        h_x_        = (float*)malloc( Nx_ * sizeof(float));
        h_x_sample_ = (float*)malloc( N_ * Nx_ * sizeof(float));
        h_dist_     = (float*)malloc( N_ * sizeof(float));
        h_w_        = (float*)malloc( N_ * sizeof(float));

        // dans le constructeur, après les malloc
        std::normal_distribution<float> dist_x(150.0f, 50.0f);  
        std::normal_distribution<float> dist_y(100.0f, 50.0f);
        for(int i = 0; i < N_; i++) {
            h_x_sample_[i * Nx_ + 0] = dist_x(gen);
            h_x_sample_[i * Nx_ + 1] = dist_y(gen);
            h_w_[i] = 1.0f / N_;
        }
        cudaMemcpy(d_x_sample_, h_x_sample_, N_ * Nx_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w_, h_w_, N_ * sizeof(float), cudaMemcpyHostToDevice);

        min_dist_ = 1e6f;

    }

    sir::~sir(){
        free(h_x_);
        free(h_x_sample_);
        free(h_dist_);
        free(h_w_);

        cudaFree(d_z_sample_);
        cudaFree(d_x_sample_);
        cudaFree(d_w_);
        cudaFree(d_z_);
        cudaFree(d_dist_);
        cudaFree(d_states_);
    }

    void sir::normalization() { //compute of the weights according to the likelyhood as the formula of the particle filter shows
        float sum_w = 0.0f;
        for(int i = 0; i < N_; i++) sum_w += h_w_[i];
        if(sum_w == 0.0f || std::isnan(sum_w)) {
            for(int i = 0; i < N_; i++) h_w_[i] = 1.0f / N_;
            return;
        }
        for(int i = 0; i < N_; i++) h_w_[i] /= sum_w; 
    }



    float sir::computeESS() { //ess as proposed in the Arulampalam's article which is used to create a condition for the resampling
        float sum_sq = 0.0f;
        for(int i = 0; i < N_; i++) sum_sq += h_w_[i] * h_w_[i];
        return 1.0f / sum_sq;
    }
    
    

    void sir::estimation(){ //estimation of the probability distribution
        for (int k = 0 ; k < Nx_ ; k++){
            h_x_[k] = 0;
        }  

        for(int i = 0; i < N_ ; i ++){
            for (int k = 0 ; k < Nx_ ; k++){
                    h_x_[k] += h_x_sample_[i*Nx_ + k] * h_w_[i];
            }  
        }
    }




    void sir::resampling(std::mt19937& gen) {
        //Compute the Cumulative Distribution Function (CDF)
        std::vector<float> cdf(N_);
        cdf[0] = h_w_[0];
        for (int l = 1; l < N_; l++) {
            cdf[l] = cdf[l - 1] + h_w_[l];
        }

        //Setup the systematic selection
        std::uniform_real_distribution<float> dis(0.0f, 1.0f / N_);
        float u = dis(gen); 
        
        std::vector<float> new_samples(Nx_ * N_);
        int i = 0;

        // The Wheel Selection
        for (int j = 0; j < N_; j++) {
            float uj = u + (float)j / N_; // The current pointer on the wheel
            
            while (uj > cdf[i] && i < N_ - 1) {
                i++; // Advance in the CDF until we find the interval containing uj
            }
            
            //Copy the selected particle i into the new population at index j
            for (int k = 0; k < Nx_; k++) {
                new_samples[j * Nx_ + k] = h_x_sample_[i * Nx_ + k];
            }
        }

        // Final Step: Replace the old population
        memcpy(h_x_sample_, new_samples.data(), Nx_ * N_ * sizeof(float));
        
        // Reset weights to uniform (all particles are now equally probable)
        for(int i = 0; i < N_; i++) h_w_[i] = 1.0f / N_;
    }

    void sir::run_smc(std::mt19937& gen) {
        dim3 threadsPerBlock(256);
        dim3 grid((N_ + threadsPerBlock.x - 1) / threadsPerBlock.x);

        // 1. Predict
        prediction_kernel<<<grid, threadsPerBlock>>>(d_states_, d_x_sample_, Nx_, N_);
        cudaDeviceSynchronize();

        // 2. Sample measurements from fluid
        sample_velocity_kernel<<<grid, threadsPerBlock>>>(myFluid_.getDevice_ux_(),
                                                        myFluid_.getDevice_uy_(),
                                                        d_z_, Nz_/2,
                                                        myFluid_.getGridDim()[0]);
        cudaDeviceSynchronize();

        // 3. Compute distances and raw weights
        weigthUpdate<<<grid, threadsPerBlock>>>(N_, Nx_, Nz_, a_, u0_, min_dist_,
                                                d_z_, d_x_sample_, d_z_sample_,
                                                d_dist_, d_w_);
        cudaDeviceSynchronize();

        // 4. Get min distance using host copy (simple and stable)
        cudaMemcpy(h_dist_, d_dist_, N_ * sizeof(float), cudaMemcpyDeviceToHost);
        min_dist_ = *std::min_element(h_dist_, h_dist_ + N_);

        // 5. Copy weights to host for normalization & ESS
        cudaMemcpy(h_w_, d_w_, N_ * sizeof(float), cudaMemcpyDeviceToHost);

        // 6. Normalize host weights
        normalization();

        // 7. ESS & resampling
        if (computeESS() < N_ * 0.5f) {
            resampling(gen);
            cudaMemcpy(d_x_sample_, h_x_sample_, N_ * Nx_ * sizeof(float), cudaMemcpyHostToDevice);
            // weights already uniform on host, copy back to device
            cudaMemcpy(d_w_, h_w_, N_ * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            // Copy normalized weights back to device for next iteration
            cudaMemcpy(d_w_, h_w_, N_ * sizeof(float), cudaMemcpyHostToDevice);
        }

        // 8. Estimate state from host arrays
        estimation();   // fills h_x_

        // No need for d_x_ copy unless you use it elsewhere
    }

}

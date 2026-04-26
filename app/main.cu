#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include "../include/lbm/lbm_gpu.h"
#include "../renderer/render_gpu.h"
#include "../include/filter/smc_gpu.h"
#include "../utils/metrics.h"

int main(){
        // --- basic parameters ---
    int N = 1000000;
    int Nx = 2;
    int Nz = 10;
    int w = 500, h = 200;
    float u0 = 0.1f;
    float visc = 0.04f;
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr))); // must be initialized in main



    Physics::lbm sim(2, 9, h, w, visc, u0);  //LBM constructor 

    Display display(w,h); //renderer contructor
    sim.runInit(100, 80, 40); // initial condition of the fluid 
    cudaDeviceSynchronize();

    smc::sir filter(sim, N, Nx, Nz, 3, u0, gen);  //SCM construtor

    Metrics::metrics meter(w, h, 5000, 100, u0, sim);

    auto start = std::chrono::high_resolution_clock::now();
    int t = 0;
    while (display.isOpen()) {
        sim.runCollision();
        sim.runPullStreaming();
        display.render(sim.getDeviceSpeed(), sim.getDeviceBar());
        

        if (t % 30 == 0) {  
            filter.run_smc(gen);
            
            const float* pos_real = sim.getObstaclePos();
            float* pos_est = filter.getState();
            meter.strouhalNumCompute(t);
            printf("Real: [%.1f, %.1f] | Estimated: [%.1f, %.1f] | Error: %.2f px\n", 
                pos_real[0], pos_real[1], 
                pos_est[0],  pos_est[1],
                sqrtf(powf(pos_real[0]-pos_est[0], 2.0f) + powf(pos_real[1]-pos_est[1], 2.0f)));
        }
        t++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(end-start).count();
    std::cout << "Performance: " << meter.MLUPS(t, ms) << " MLUPS" << std::endl;
}
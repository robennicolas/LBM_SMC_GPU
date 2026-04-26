#include "metrics.h"

namespace Metrics {

metrics::metrics(int w, int h, int warmup_steps, int xtop, float u0, Physics::lbm& sim)
    : w_(w), h_(h), warmup_steps_(warmup_steps), u0_(u0), sim_(sim),
      uy_prev_(0.0f), last_crossing_(0), crossing_count_(0), freq_sum_(0.0f) {
    // probe position: 30 cells downstream of obstacle's left edge, at mid‑height
    int mid_y = h / 2;
    probe_idx_ = (xtop + 30) + mid_y * w_;
}

void metrics::strouhalNumCompute(int t) {
    float uy_probe;
    cudaMemcpy(&uy_probe, sim_.getDevice_uy_() + probe_idx_, sizeof(float), cudaMemcpyDeviceToHost);

    if (uy_prev_ * uy_probe < 0.0f && t > warmup_steps_) {
        if (last_crossing_ > 0) {
            int period_steps = 2 * (t - last_crossing_);  // full period between two same‑sign zero crossings
            float freq = 1.0f / period_steps;            // frequency (1/timestep)
            freq_sum_ += freq;
            crossing_count_++;
            float St = (freq_sum_ / crossing_count_) * 40.0f / u0_;  // 40 = obstacle height (yheight)
            std::cout << "Step " << t << " | St = " << St << std::endl;
        }
        last_crossing_ = t;
    }
    uy_prev_ = uy_probe;
}

float metrics::MLUPS(int iterations, double elapsed_ms) {
    double total_updates = static_cast<double>(w_) * h_ * iterations * 9;
    return static_cast<float>(total_updates / (elapsed_ms * 1e-3) / 1e6);
}

} // namespace Metrics
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Nucleus sampling. Softmax probabilities are computed with parallel reductions;
// the nucleus is found by binary-searching a probability threshold t such that
// {v : p_v >= t} is the smallest prefix of the descending order with total mass >= p
// (equivalent to sorting for distinct probabilities). Sampling walks tokens in
// index order, which is a valid categorical draw over the nucleus.
__global__ void top_p_kernel(const float* logits, const float* p, const int* seed,
                             int* sampled_token, int vocab_size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // max logit
    float m = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) m = fmaxf(m, logits[i]);
    sdata[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_logit = sdata[0];
    __syncthreads();

    // sum of exponentials
    float sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) sum += expf(logits[i] - max_logit);
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float Z = sdata[0];
    __syncthreads();

    if (tid != 0) return;

    float target = *p;
    // binary search on the threshold: mass(t) = sum of probs >= t is decreasing in t
    float lo = 0.0f, hi = 1.0f;
    for (int it = 0; it < 40; it++) {
        float mid = 0.5f * (lo + hi);
        float mass = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float pv = expf(logits[v] - max_logit) / Z;
            if (pv >= mid) mass += pv;
        }
        if (mass >= target) {
            lo = mid;  // can raise the threshold further
        } else {
            hi = mid;
        }
    }
    float threshold = lo;

    float nucleus_mass = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        float pv = expf(logits[v] - max_logit) / Z;
        if (pv >= threshold) nucleus_mass += pv;
    }

    // uniform sample in [0, 1) from the seed (splitmix-style hash)
    unsigned long long s64 = (unsigned long long)(unsigned int)(*seed) + 0x9E3779B97F4A7C15ull;
    s64 = (s64 ^ (s64 >> 30)) * 0xBF58476D1CE4E5B9ull;
    s64 = (s64 ^ (s64 >> 27)) * 0x94D049BB133111EBull;
    s64 = s64 ^ (s64 >> 31);
    float r = (float)((s64 >> 11) * (1.0 / 9007199254740992.0));  // [0,1)

    float pick = r * nucleus_mass;
    float cum = 0.0f;
    int chosen = -1;
    for (int v = 0; v < vocab_size; v++) {
        float pv = expf(logits[v] - max_logit) / Z;
        if (pv >= threshold) {
            cum += pv;
            chosen = v;
            if (cum >= pick) break;
        }
    }
    *sampled_token = chosen;
}

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
    top_p_kernel<<<1, 256>>>(logits, p, seed, sampled_token, vocab_size);
    cudaDeviceSynchronize();
}

#include "roundingStuff.h"


void rounding_operation_ran() {
    alignas(16) float valA[4] = { 2, 0, 0, 0 }; // Ausgerichtet auf 16 Bytes, um Speicherprobleme zu vermeiden
    alignas(16) float valB[4] = { 10, 0, 0, 0 };

    __m128 xmm1 = _mm_loadu_ps(valA);
    __m128 xmm2 = _mm_loadu_ps(valB);

    for (int i = 0; i < MAX_IT; i++) {
        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        __m128 xmm_result = _mm_div_ps(xmm1, xmm2);
        float result[1];
        _mm_storeu_ps(result, xmm_result);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);
    }
    
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto start = high_resolution_clock::now();

    std::vector<std::thread> threads;

    for (int i = 0; i < MAX_THREAD; i++) {
        threads.push_back(std::thread(rounding_operation_ran));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stop = high_resolution_clock::now();
    auto exec_duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Execution time with random rounding: " << exec_duration.count() << " ms" << std::endl;
    std::cout << std::setprecision(128)  << "Average time per calculation: " << (exec_duration.count() * 1000 * 1000) / (MAX_IT * MAX_THREAD) << " ns" << std::endl;
    std::cout << "--------" << std::endl;

}
#include "roundingStuff.h"


void rounding_operation_fix() {
    alignas(16) float valA[4] = { 2, 0, 0, 0 }; // Ausgerichtet auf 16 Bytes, um Speicherprobleme zu vermeiden
    alignas(16) float valB[4] = { 10, 0, 0, 0 };

    __m128 xmm1 = _mm_loadu_ps(valA);
    __m128 xmm2 = _mm_loadu_ps(valB);

    for (int i = 0; i < MAX_IT; i++) {
        __m128 xmm_result = _mm_div_ps(xmm1, xmm2);
        float result[1];
        _mm_storeu_ps(result, xmm_result);

        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);

        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);

        xmm_result = _mm_div_ps(xmm1, xmm2);
        _mm_storeu_ps(result, xmm_result);
    }
    
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;


    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

    for (int i = 0; i < MAX_THREAD; i++) {
        threads.push_back(std::thread(rounding_operation_fix));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stop = high_resolution_clock::now();
    auto exec_duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Execution time with fixed rounding: " << exec_duration.count() << " ms" << std::endl ;
    std::cout << std::setprecision(128) << "Average time per calculation: " << (exec_duration.count() * 1000 * 1000) / (MAX_IT * MAX_THREAD) << " ns" << std::endl;
    std::cout << "--------" << std::endl;

    return 0;
}
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include <xmmintrin.h>  // SSE header
#include <vector>
#include <cstdlib>
#include <chrono>

// Number of iterations per thread
int MAX_IT = 10000;
// Number of threads
int MAX_THREAD = 10000;


// Execute a randomly rounded operation (2 divided by 10)
void rounding_operation(int i) {
    alignas(16) float valA[4] = { 2, 0, 0, 0 }; // Ausgerichtet auf 16 Bytes, um Speicherprobleme zu vermeiden
    alignas(16) float valB[4] = { 10, 0, 0, 0 };

    __m128 xmm1 = _mm_loadu_ps(valA);
    __m128 xmm2 = _mm_loadu_ps(valB);

    for (int i = 0; i < MAX_IT; i++) {
        int mode = rand() % 4;

        switch (mode) {
         case 0: _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN); 
        break;
         case 1: _MM_SET_ROUNDING_MODE(_MM_ROUND_UP); 
        break;
         case 2: _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST); 
        break;
         case 3: _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO); 
        break;
        }

        __m128 xmm_result = _mm_div_ps(xmm1, xmm2);

        float result[1];

        _mm_storeu_ps(result, xmm_result);

        if (mode == 0 || mode == 3) {
            if (result[0] > 0.2) {
                std::cout << "Rounding error" << std::endl;
            }
        }

        if (mode == 1 || mode == 2) {
            if (result[0] < 0.2) {
                std::cout << "Rounding error" << std::endl;
            }
        }
    }
} 


int main() {
    // Multithreaded calculation
    std::vector<std::thread> threads;

    for (int i = 0; i < MAX_THREAD; i++) {
        threads.push_back(std::thread(rounding_operation, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}

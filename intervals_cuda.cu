#include "Header.h"

#include <iostream>
#include <functional>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <chrono>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

using namespace std;

// Primary functions
interval<double> withLoop(interval<double> x, string w, function<void(const interval<double>&, interval<double>*)> f);
interval<double> withRecursion(interval<double> x, double w, function<void(const interval<double>&, interval<double>*)> f);

vector<interval<double>> calculateIntervalWidth(interval<double> x, string w);

// Host-Funktion zur Ausführung des CUDA-Kernels
interval<double> processInterval(interval<double> x, double w) {
    interval<double>* cuda_result;
    interval<double> result;

    cudaMalloc(&cuda_result, sizeof(interval<double>));
    int blockSize = 1024;
    int numBlocks = 1;
    int sharedMemSize = blockSize * sizeof(interval<double>);

    processIntervalKernel << <numBlocks, blockSize, sharedMemSize >> > (x, w, cuda_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, cuda_result, sizeof(interval<double>), cudaMemcpyDeviceToHost);
    cudaFree(cuda_result);

    return result;
}

int main() {

    interval<double> x(-1, 3);
    double w = 0.01;

    auto start = chrono::high_resolution_clock::now();

    auto result = processInterval(x, w);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "CUDA Result: [" << result.lower() << ", " << result.upper() << "]" << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl << "===========================" << endl;

    return 0;
}

/*
int main() {

    
    auto f = [](const interval<double>& x, interval<double>* r) -> void {

        auto result_a = mulIntervals(powIntervals(x, 3), interval<double>(2, 2));

        auto result_b = mulIntervals(interval<double>(3, 3), powIntervals(x, 2));

        auto result_c = mulIntervals(interval<double>(2, 2), x);

        auto result = addIntervals(subIntervals(addIntervals(result_a, result_b), result_c), interval<double>(1, 1));
        
        r->assign(result.lower(), result.upper());
        return;
        };

    interval<double> x(-1, 1);
    interval<double> y(3, 12);

    auto test = x.cross(y);

    cout << "Loop result: [" << test.lower() << ", " << test.upper() << "]" << endl;

    auto asdf = calculateIntervalWidth(x, "0.5");

    for (int i = 0; i < asdf.size(); i++) {

    }

    /*


    string w = "0.1";

    
    auto start = chrono::high_resolution_clock::now();

    interval<double> loopResult = withLoop(x, w, f);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Loop result: [" << loopResult.lower() << ", " << loopResult.upper() << "]" << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl << "===========================" << endl;



    start = chrono::high_resolution_clock::now();

    interval<double> recursiveResult = withRecursion(x, 0.1, f);

    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Recursive result: [" << recursiveResult.lower() << ", " << recursiveResult.upper() << "]" << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl;

    return 0;
    

    /*
    interval<double> add = addIntervals(a, b);
    std::cout << "Result add: [" << add.lower() << ", " << add.upper() << "]" << std::endl;

    interval<double> sub = subIntervals(a, b);
    std::cout << "Result sub: [" << sub.lower() << ", " << sub.upper() << "]" << std::endl;

    interval<double> mul = mulIntervals(a, b);
    std::cout << "Result mul: [" << mul.lower() << ", " << mul.upper() << "]" << std::endl;
    

    //interval<double> divD = divIntervals(a, b);
    //std::cout << setprecision(32) << "Result div: [" << divD.lower() << ", " << divD.upper() << "]" << std::endl;

    //interval<float> divF = divIntervals(c, d);
    //std::cout << setprecision(32) << "Result div: [" << divF.lower() << ", " << divF.upper() << "]" << std::endl;

    


    return 0;
}

vector<interval<double>> calculateIntervalWidth(interval<double> x, string w) {
    // First convert double to int, to combat rounding floating point errors:

    int scale = 1;
    double w_val = stod(w);

    // Only neccessary when there actually are decimal places
    if ((int)w_val != w_val) {
        int lenAfterDot = w.length();
        int i = 0;

        // Check length of decimal places
        for (lenAfterDot; lenAfterDot > 0; lenAfterDot--) {

            if (w[i] == '.') {
                break;
            }
            i++;
        }

        // Scale value so we only have whole numbers
        scale = pow(10, lenAfterDot - 1);
        w_val *= scale;
    }

    //cout << w_val << " - " << scale << endl;

    double scaled_width = x.width() * scale;

    // Calculate how many intervals fit
    double needed_intervals_calc = scaled_width / w_val;
    int needed_intervals = (int)needed_intervals_calc + ((int)needed_intervals_calc != needed_intervals_calc ? 1 : 0);

    double target_interval_size = scaled_width / needed_intervals;

    //cout << target_interval_size << endl;

    // Build vector of all sub-intervals
    vector<interval<double>> split_results = { interval<double>(x.lower() * scale, x.lower() * scale + target_interval_size) };

    for (int i = 1; i < needed_intervals; i++) {
        split_results.push_back(interval<double>(split_results[i - 1].upper(), split_results[i - 1].upper() + target_interval_size));
        cout << split_results[i].lower() << ", " << split_results[i].upper() << endl;
    }

    vector<interval<double>> result;

    for (int i = 0; i < split_results.size(); i++) {
        result.push_back(interval<double>(split_results[i].lower() / scale, split_results[i].upper() / scale));
    }

    return result;
}

/*
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int mainold()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/
//https://gist.github.com/dpiponi/1502434
/*
https://www.sciencedirect.com/science/article/abs/pii/B9780123859631000095
https://hal.science/hal-00263670/document
https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE
*/



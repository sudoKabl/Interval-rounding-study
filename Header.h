#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<class T>
class interval
{
	public:
		__device__ __host__ interval() : low(0), up(0) {}
		__device__ __host__ interval(T const& v) : low(v), up(v) {}
		__device__ __host__ interval(T const& l, T const& u) : low(l), up(u) {}
		__device__ __host__ T const& lower() const { return low; }
		__device__ __host__ T const& upper() const { return up; }
		__device__ __host__ void assign(T l, T u) { this->low = l; this->up = u; return; }
		__device__ __host__ T width() { return up - low; }
		__device__ __host__ T median() { return (((up - low) / 2) + low); }
		
		__device__ __host__ interval<T> cross(interval<T> other) {
			T lower = (this->low < other.lower() ? this->low : other.lower());
			T upper = (this->up > other.upper() ? this->up : other.upper());
			return interval<T>(lower, upper);
		}


		static __device__ __host__ interval empty() { return interval(T(0), T(0)); }
	private:
		T low;
		T up;
};


template<class T>
struct rounded_arith
{
	__device__ T add_down(const T& x, const T& y);
	__device__ T add_up(const T& x, const T& y);
	__device__ T sub_down(const T& x, const T& y);
	__device__ T sub_up(const T& x, const T& y);
	__device__ T mul_down(const T& x, const T& y);
	__device__ T mul_up(const T& x, const T& y);
	__device__ T div_down(const T& x, const T& y);
	__device__ T div_up(const T& x, const T& y);
	__device__ T median(const T& x, const T& y);
	__device__ T sqrt_down(const T& x);
	__device__ T sqrt_up(const T& x);
	__device__ T int_down(const T& x);
	__device__ T int_up(const T& x);
	__device__ T pow_any(const T& x, const T& p);
};


// --- Addition ---

template<class T> inline __device__
interval<T> operator+(interval<T> const& x, interval<T> const& y)
{
	rounded_arith<T> rnd;
	T l = rnd.add_down(x.lower(), y.lower());
	T u = rnd.add_up(x.upper(), y.upper());
	return interval<T>(min(l, u), max(l, u));
}

// --- Subtraction ---

template<class T> inline __device__
interval<T> operator-(interval<T> const& x, interval<T> const& y)
{
	rounded_arith<T> rnd;
	T l = rnd.sub_down(x.lower(), y.upper());
	T u = rnd.sub_up(x.upper(), y.lower());
	return interval<T>(min(l, u), max(l, u));
}

// --- Multiplication ---

template<class T> inline __device__
interval<T> operator*(interval<T> const& x, interval<T> const& y)
{
	rounded_arith<T> rnd;

	T min_lower = min(
		min(rnd.mul_down(x.lower(), y.lower()),
			rnd.mul_down(x.lower(), y.upper())),
		min(rnd.mul_down(x.upper(), y.lower()),
			rnd.mul_down(x.upper(), y.upper()))
		);

	T max_upper = max(
		max(rnd.mul_up(x.lower(), y.lower()),
			rnd.mul_up(x.lower(), y.upper())),
		max(rnd.mul_up(x.upper(), y.lower()),
			rnd.mul_up(x.upper(), y.upper()))
		);

	return interval<T>(min_lower, max_upper);
}

// --- Divison ---

template<class T> inline __device__
interval<T> operator/(interval<T> const& x, interval<T> const& y)
{
	rounded_arith<T> rnd;
	interval<T> inverse( rnd.div_down(1, y.upper()), rnd.div_up(1, y.lower()) );
	return interval<T>(x * inverse);
}

template<class T> inline __device__
interval<T> devicePow(const interval<T>& x, int pwr)
{
	if (pwr == 0) {
		interval<T> e;
		return e.empty();
	}

	rounded_arith<T> rnd;

	
	if (pwr < 0) {
		return interval<T>(1, 1) / devicePow(x, -pwr);
	}

	if (x.upper() < 0) { // Both negative
		T yl = rnd.pow_any(-x.upper(), pwr);
		T yu = rnd.pow_any(-x.lower(), pwr);

		if (pwr % 2 == 1) 
			return interval<T>(min(-yu, -yl), max(-yu, -yl));
		else            
			return interval<T>(min(yl, yu), max(yl, yu));
	}
	else if (x.lower() < 0) { // Only lower negative
		if (pwr & 1) {   // [-1,1]^1
			return interval<T>(-rnd.pow_any(-x.lower(), pwr), rnd.pow_any(x.upper(), pwr));
		}
		else {         // [-1,1]^2
			return interval<T>(0, rnd.pow_any(max(-x.lower(), x.upper()), pwr));
		}
	}
	else {
		return interval<T>(rnd.pow_any(x.lower(), pwr), rnd.pow_any(x.upper(), pwr));
	}
}

// ------------------------------------------------------------------------------------------------
// ----- Float declarations
// ------------------------------------------------------------------------------------------------

template <>
struct rounded_arith<float>
{
	
	__device__ float add_down(const float& x, const float& y) {

		return __fadd_rd(x, y);
	}

	__device__ float add_up(const float& x, const float& y) {
		return __fadd_ru(x, y);
	}

	__device__ float sub_down(const float& x, const float& y) {

		return __fsub_rd(x, y);
	}

	__device__ float sub_up(const float& x, const float& y) {
		return __fsub_ru(x, y);
	}

	__device__ float mul_down(const float& x, const float& y) {
		return __fmul_rd(x, y);
	}

	__device__ float mul_up(const float& x, const float& y) {
		return __fmul_ru(x, y);
	}

	__device__ float div_down(const float& x, const float& y) {
		return __fdiv_rd(x, y);
	}

	__device__ float div_up(const float& x, const float& y) {
		return __fdiv_ru(x, y);
	}

	__device__ float pow_any(const float& x, const int& p) {
		return pow(x, p);
	}
};



typedef void(*KernelFunc)(interval<float>, interval<float>, interval<float>*);

__global__ void addIntervalsKernel(interval<float> a, interval<float> b, interval<float>* result) {
	*result = a + b;
}


__global__ void subIntervalsKernel(interval<float> a, interval<float> b, interval<float>* result) {
	*result = a - b;
}


__global__ void mulIntervalsKernel(interval<float> a, interval<float> b, interval<float>* result) {
	*result = a * b;
}


__global__ void divIntervalsKernel(interval<float> a, interval<float> b, interval<float>* result) {
	*result = a / b;
}

__global__ void powIntervalsKernel(interval<float> a, int pow, interval<float>* result) {
	*result = devicePow(a, pow);
}


interval<float> executeIntervalOperation(KernelFunc func, interval<float> a, interval<float> b) {
	interval<float>* cuda_result;
	interval<float> result;

	// Allocate memory on gpu
	cudaMalloc(&cuda_result, sizeof(interval<float>));

	// Add intervalls using cuda

	func << <1, 1 >> > (a, b, cuda_result);
	cudaDeviceSynchronize();

	// Copy result back and return result
	cudaMemcpy(&result, cuda_result, sizeof(interval<float>), cudaMemcpyDeviceToHost);

	return result;
}


interval<float> addIntervals(interval<float> a, interval<float> b) {
	return executeIntervalOperation(addIntervalsKernel, a, b);
}


interval<float> subIntervals(interval<float> a, interval<float> b) {
	return executeIntervalOperation(subIntervalsKernel, a, b);
}


interval<float> mulIntervals(interval<float> a, interval<float> b) {
	return executeIntervalOperation(mulIntervalsKernel, a, b);
}


interval<float> divIntervals(interval<float> a, interval<float> b) {
	return executeIntervalOperation(divIntervalsKernel, a, b);
}

interval<float> powIntervals(interval<float> a, int pow) {
	interval<float>* cuda_result;
	interval<float> result;

	cudaMalloc(&cuda_result, sizeof(interval<float>));

	powIntervalsKernel << <1, 1 >> > (a, pow, cuda_result);
	cudaDeviceSynchronize();

	cudaMemcpy(&result, cuda_result, sizeof(interval<float>), cudaMemcpyDeviceToHost);

	return result;
}


// ------------------------------------------------------------------------------------------------
// ----- Double declarations
// ------------------------------------------------------------------------------------------------


template <>
struct rounded_arith<double>
{

	__device__ double add_down(const double& x, const double& y) {

		return __dadd_rd(x, y);
	}

	__device__ double add_up(const double& x, const double& y) {
		return __dadd_ru(x, y);
	}

	__device__ double sub_down(const double& x, const double& y) {

		return __dsub_rd(x, y);
	}

	__device__ double sub_up(const double& x, const double& y) {
		return __dsub_ru(x, y);
	}

	__device__ double mul_down(const double& x, const double& y) {
		return __dmul_rd(x, y);
	}

	__device__ double mul_up(const double& x, const double& y) {
		return __dmul_ru(x, y);
	}

	__device__ double div_down(const double& x, const double& y) {
		return __ddiv_rd(x, y);
	}

	__device__ double div_up(const double& x, const double& y) {
		return __ddiv_ru(x, y);
	}

	__device__ double pow_any(const double& x, const int& p) {
		return pow(x, p);
	}
};



typedef void(*KernelFuncDouble)(interval<double>, interval<double>, interval<double>*);

__global__ void addIntervalsKernel(interval<double> a, interval<double> b, interval<double>* result) {
	*result = a + b;
}


__global__ void subIntervalsKernel(interval<double> a, interval<double> b, interval<double>* result) {
	*result = a - b;
}


__global__ void mulIntervalsKernel(interval<double> a, interval<double> b, interval<double>* result) {
	*result = a * b;
}


__global__ void divIntervalsKernel(interval<double> a, interval<double> b, interval<double>* result) {
	*result = a / b;
}


__global__ void powIntervalsKernel(interval<double> a, int pow, interval<double>* result) {
	*result = devicePow(a, pow);
}


interval<double> executeIntervalOperationDouble(KernelFuncDouble func, interval<double> a, interval<double> b) {
	interval<double>* cuda_result;
	interval<double> result;

	// Allocate memory on gpu
	cudaMalloc(&cuda_result, sizeof(interval<double>));

	// Add intervalls using cuda

	func << <1, 1 >> > (a, b, cuda_result);
	cudaDeviceSynchronize();

	// Copy result back and return result
	cudaMemcpy(&result, cuda_result, sizeof(interval<double>), cudaMemcpyDeviceToHost);

	return result;
}


interval<double> addIntervals(interval<double> a, interval<double> b) {
	return executeIntervalOperationDouble(addIntervalsKernel, a, b);
}


interval<double> subIntervals(interval<double> a, interval<double> b) {
	return executeIntervalOperationDouble(subIntervalsKernel, a, b);
}


interval<double> mulIntervals(interval<double> a, interval<double> b) {
	return executeIntervalOperationDouble(mulIntervalsKernel, a, b);
}


interval<double> divIntervals(interval<double> a, interval<double> b) {
	return executeIntervalOperationDouble(divIntervalsKernel, a, b);
}


interval<double> powIntervals(interval<double> a, int pow) {
	interval<double>* cuda_result;
	interval<double> result;

	cudaMalloc(&cuda_result, sizeof(interval<double>));

	powIntervalsKernel<< <1, 1 >> > (a, pow, cuda_result);
	cudaDeviceSynchronize();

	cudaMemcpy(&result, cuda_result, sizeof(interval<double>), cudaMemcpyDeviceToHost);

	return result;
}




/*
// ------------------------------------------------------------------------------------------------
// ----- Functional
// ------------------------------------------------------------------------------------------------
*/

// Funktion zur Unterteilung eines Intervalls in kleinere Intervalle auf der GPU
__device__ void splitInterval(const interval<double>& x, double w, interval<double>* subIntervals, int* count) {
	double lower = x.lower();
	double upper = x.upper();
	int index = 0;

	while (lower < upper) {
		double nextUpper = fmin(lower + w, upper);
		subIntervals[index++] = interval<double>(lower, nextUpper);
		lower = nextUpper;
	}

	*count = index;
}

// CUDA Kernel zum Anwenden einer Funktion auf ein Intervall
__device__ void applyFunction(const interval<double>& x, interval<double>* result) {
	*result = (interval<double>(2, 2) * devicePow(x, 3)) +
		(interval<double>(3, 3) * devicePow(x, 2)) +
		(interval<double>(2, 2) * x) +
		interval<double>(1, 1);
}

// Funktion zur Vereinigung zweier Intervalle auf der GPU
__device__ interval<double> unionIntervals(const interval<double>& a, const interval<double>& b) {
	double lower = fmin(a.lower(), b.lower());
	double upper = fmax(a.upper(), b.upper());
	return interval<double>(lower, upper);
}

// Hauptkernel zur Unterteilung, Anwendung der Funktion und Vereinigung auf der GPU
__global__ void processIntervalKernel(interval<double> x, double w, interval<double>* result) {
	__shared__ interval<double> subIntervals[1024]; // Annahme: max 1024 Unterintervalle
	__shared__ int count;

	if (threadIdx.x == 0) {
		splitInterval(x, w, subIntervals, &count);
	}
	__syncthreads();

	if (threadIdx.x < count) {
		interval<double> tempResult;
		applyFunction(subIntervals[threadIdx.x], &tempResult);
		subIntervals[threadIdx.x] = tempResult;
	}
	__syncthreads();

	// Parallel reduction to union all intervals
	for (int stride = 1; stride < count; stride *= 2) {
		int index = 2 * stride * threadIdx.x;
		if (index < count) {
			subIntervals[index] = unionIntervals(subIntervals[index], subIntervals[index + stride]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		*result = subIntervals[0];
	}
}

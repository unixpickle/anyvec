extern "C" __global__ void divElements(float * x, float * y, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] /= y[tid];
	}
}

extern "C" __global__ void expElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = expf(x[tid]);
	}
}

extern "C" __global__ void tanhElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = tanhf(x[tid]);
	}
}

extern "C" __global__ void sinElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = sinf(x[tid]);
	}
}

extern "C" __global__ void clipPositive(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = fmaxf(0, x[tid]);
	}
}

extern "C" __global__ void shiftRandUniform(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] == 1.0f) {
			x[tid] = 0;
		}
	}
}

extern "C" __global__ void uniformToBernoulli(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] > 0.5) {
			x[tid] = 1;
		} else {
			x[tid] = 0;
		}
	}
}

extern "C" __global__ void addRepeated(float * dest, float * source, size_t destLen, size_t sourceLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid % sourceLen];
	}
}

extern "C" __global__ void addRepeatedPow2(float * dest, float * source, size_t destLen, size_t srcMask) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid & srcMask];
	}
}

extern "C" __global__ void addScaler(float s, float * dest, size_t destLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += s;
	}
}

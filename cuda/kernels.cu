extern "C" __global__
void divElements(float * x, float * y, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] /= y[tid];
	}
}

extern "C" __global__
void expElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = expf(x[tid]);
	}
}

extern "C" __global__
void logElements(float * x, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = logf(x[tid]);
	}
}

extern "C" __global__
void tanhElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = tanhf(x[tid]);
	}
}

extern "C" __global__
void sinElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = sinf(x[tid]);
	}
}

extern "C" __global__
void clipPositive(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = fmaxf(0, x[tid]);
	}
}

extern "C" __global__
void shiftRandUniform(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] == 1.0f) {
			x[tid] = 0;
		}
	}
}

extern "C" __global__
void uniformToBernoulli(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] > 0.5) {
			x[tid] = 1;
		} else {
			x[tid] = 0;
		}
	}
}

extern "C" __global__
void addRepeated(float * dest, float * source, size_t destLen, size_t sourceLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid % sourceLen];
	}
}

extern "C" __global__
void addRepeatedPow2(float * dest, float * source, size_t destLen, size_t srcMask) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid & srcMask];
	}
}

extern "C" __global__
void scaleRepeated(float * dest, float * source, size_t destLen, size_t sourceLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] *= source[tid % sourceLen];
	}
}

extern "C" __global__
void scaleRepeatedPow2(float * dest, float * source, size_t destLen, size_t srcMask) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] *= source[tid & srcMask];
	}
}

extern "C" __global__
void addScaler(float s, float * dest, size_t destLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += s;
	}
}

extern "C" __global__
void addChunks(float * dest, float * source, size_t destLen, size_t chunkSize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid / chunkSize];
	}
}

extern "C" __global__
void subChunks(float * dest, float * source, size_t destLen, size_t chunkSize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] -= source[tid / chunkSize];
	}
}

extern "C" __global__
void lessThan(float s, float * v, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
    if (v[tid] < s) {
      v[tid] = 1;
    } else {
      v[tid] = 0;
    }
	}
}

extern "C" __global__
void greaterThan(float s, float * v, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
    if (v[tid] > s) {
      v[tid] = 1;
    } else {
      v[tid] = 0;
    }
	}
}

extern "C" __global__
void equalTo(float s, float * v, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
    if (v[tid] == s) {
      v[tid] = 1;
    } else {
      v[tid] = 0;
    }
	}
}

extern "C" __device__
float addLogPair(float x, float y) {
  float m = max(x, y);
  return logf(expf(x-m) + expf(y-m)) + m;
}

extern "C" __global__
void addLogs(float * dst, float * src, size_t rowSize) {
  extern __shared__ float chunk[];

  size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowIdx < rowSize) {
    chunk[threadIdx.x] = src[rowIdx+rowSize*blockIdx.y];
  }
  __syncthreads();

  for (size_t stride = (blockDim.x>>1); stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride && rowIdx+stride < rowSize) {
      chunk[threadIdx.x] = addLogPair(chunk[threadIdx.x],
        chunk[threadIdx.x+stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    dst[blockIdx.x + blockIdx.y*gridDim.x] = chunk[0];
  }
}

extern "C" __global__
void powScaler(float s, float * dest, size_t destLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] = powf(dest[tid], s);
	}
}

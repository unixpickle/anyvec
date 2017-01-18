//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvrtc.h"

const char ** nullStrPtr = NULL;
const CUjit_option * nullJitOptions = NULL;
const void ** nullPtrPtr = NULL;

void kernel_sizes(size_t n, unsigned int * block, unsigned int * grid) {
	*block = 128;
	if (n < *block) {
		*block = (unsigned int)n;
		*grid = 1;
	} else {
		*grid = (unsigned int)(n / (size_t)(*block));
		if (n%(*block) != 0) {
			(*grid)++;
		}
	}
}

CUresult anyvec_cuda_call2(CUfunction f, size_t n, void * p1, void * p2) {
	void * args[] = {&p1, &p2, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call1(CUfunction f, size_t n, void * p1) {
	void * args[] = {&p1, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call2_asym(CUfunction f, size_t n1, size_t n2, void * p1, void * p2) {
	void * args[] = {&p1, &p2, &n1, &n2};
	unsigned int blockSize, gridSize;
	kernel_sizes(n1, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call1_scaler(CUfunction f, size_t n, float scaler, void * p1) {
	void * args[] = {&scaler, &p1, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}
*/
import "C"

import (
	"math"
	"unsafe"
)

type mathKernels struct {
	module  C.CUmodule
	kernels map[string]C.CUfunction
	prog    C.nvrtcProgram
}

func newMathKernels() (kernels *mathKernels, err error) {
	code := C.CString(mathKernelsCode)
	defer C.free(unsafe.Pointer(code))
	fileName := C.CString("prog.cu")
	defer C.free(unsafe.Pointer(fileName))

	var prog C.nvrtcProgram
	res := C.nvrtcCreateProgram(&prog, code, fileName, 0,
		C.nullStrPtr, C.nullStrPtr)
	if err := nvrtcError("nvrtcCreateProgram", res); err != nil {
		return nil, err
	}

	defer func() {
		if kernels == nil {
			C.nvrtcDestroyProgram(&prog)
		}
	}()

	for _, name := range mathKernelNames {
		cName := C.CString(name)
		res = C.nvrtcAddNameExpression(prog, cName)
		C.free(unsafe.Pointer(cName))
		if err := nvrtcError("nvrtcAddNameExpression", res); err != nil {
			return nil, err
		}
	}
	res = C.nvrtcCompileProgram(prog, 0, C.nullStrPtr)
	fixSignals()
	if err := nvrtcError("nvrtcCompileProgram", res); err != nil {
		return nil, err
	}
	var ptxSize C.size_t
	res = C.nvrtcGetPTXSize(prog, &ptxSize)
	if err := nvrtcError("nvrtcGetPTXSize", res); err != nil {
		return nil, err
	}
	ptx := C.malloc(ptxSize)
	defer C.free(ptx)
	res = C.nvrtcGetPTX(prog, (*C.char)(ptx))
	if err := nvrtcError("nvrtcGetPTX", res); err != nil {
		return nil, err
	}

	var module C.CUmodule
	cuRes := C.cuModuleLoadDataEx(&module, ptx, 0, C.nullJitOptions, C.nullPtrPtr)
	if err := cuError("cuModuleLoadDataEx", cuRes); err != nil {
		return nil, err
	}
	defer func() {
		if kernels == nil {
			C.cuModuleUnload(module)
		}
	}()

	kernelMap := map[string]C.CUfunction{}
	for _, nameStr := range mathKernelNames {
		var name *C.char
		origName := C.CString(nameStr)
		defer C.free(unsafe.Pointer(origName))
		res = C.nvrtcGetLoweredName(prog, origName, &name)
		if err := nvrtcError("nvrtcGetLoweredName", res); err != nil {
			return nil, err
		}
		var kernel C.CUfunction
		cuRes := C.cuModuleGetFunction(&kernel, module, name)
		if err := cuError("cuModuleGetFunction", cuRes); err != nil {
			return nil, err
		}
		kernelMap[nameStr] = kernel
	}

	return &mathKernels{
		module:  module,
		kernels: kernelMap,
		prog:    prog,
	}, nil
}

func (m *mathKernels) Destroy() {
	C.cuModuleUnload(m.module)
	C.nvrtcDestroyProgram(&m.prog)
}

// Div32 performs element-wise division.
func (m *mathKernels) Div32(n int, num, denom unsafe.Pointer) error {
	return m.call2("divElements", n, num, denom)
}

// Exp32 performs element-wise exponentiation.
func (m *mathKernels) Exp32(n int, v unsafe.Pointer) error {
	return m.call1("expElements", n, v)
}

// Tanh32 performs element-wise hyperbolic tangent.
func (m *mathKernels) Tanh32(n int, v unsafe.Pointer) error {
	return m.call1("tanhElements", n, v)
}

// Sin32 performs element-wise sine.
func (m *mathKernels) Sin32(n int, v unsafe.Pointer) error {
	return m.call1("sinElements", n, v)
}

// ClipPos32 performs element-wise max(0, x).
func (m *mathKernels) ClipPos32(n int, v unsafe.Pointer) error {
	return m.call1("clipPositive", n, v)
}

// ShiftRandUniform32 sets all 1.0f values to 0.0f in
// order to match Go's rand package.
func (m *mathKernels) ShiftRandUniform32(n int, v unsafe.Pointer) error {
	return m.call1("shiftRandUniform", n, v)
}

// UniformToBernoulli32 creates bernoulli random variables
// from uniform random variables.
func (m *mathKernels) UniformToBernoulli32(n int, v unsafe.Pointer) error {
	return m.call1("uniformToBernoulli", n, v)
}

// AddRepeated32 adds a repeated vector to a target.
func (m *mathKernels) AddRepeated32(dstLen, srcLen int, dst, src unsafe.Pointer) error {
	log2 := uint(math.Log2(float64(srcLen)))
	if (1 << log2) == srcLen {
		return m.call2Asym("addRepeatedPow2", dstLen, srcLen-1, dst, src)
	} else {
		return m.call2Asym("addRepeated", dstLen, srcLen, dst, src)
	}
}

// AddScaler32 adds a scaler to a target.
func (m *mathKernels) AddScaler32(n int, alpha float32, v unsafe.Pointer) error {
	return m.call1Scaler("addScaler", n, alpha, v)
}

func (m *mathKernels) call1(name string, n int, v unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call1(k, C.size_t(n), v))
}

func (m *mathKernels) call1Scaler(name string, n int, s float32, v unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call1_scaler(k, C.size_t(n), C.float(s), v))
}

func (m *mathKernels) call2(name string, n int, v1, v2 unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call2(k, C.size_t(n), v1, v2))
}

func (m *mathKernels) call2Asym(name string, n1, n2 int, v1, v2 unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call2_asym(k, C.size_t(n1), C.size_t(n2), v1, v2))
}

func (m *mathKernels) doneKernel(res C.CUresult) error {
	if err := cuError("cuLaunchKernel", res); err != nil {
		return err
	}
	return m.sync()
}

func (m *mathKernels) sync() error {
	return cuError("cuCtxSynchronize", C.cuCtxSynchronize())
}

var mathKernelNames = []string{"divElements", "expElements", "tanhElements",
	"sinElements", "clipPositive", "shiftRandUniform", "uniformToBernoulli",
	"addRepeated", "addRepeatedPow2", "addScaler"}

const mathKernelsCode string = `
__global__ void divElements(float * x, float * y, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] /= y[tid];
	}
}

__global__ void expElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = expf(x[tid]);
	}
}

__global__ void tanhElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = tanhf(x[tid]);
	}
}

__global__ void sinElements(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = sinf(x[tid]);
	}
}

__global__ void clipPositive(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] = fmaxf(0, x[tid]);
	}
}

__global__ void shiftRandUniform(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] == 1.0f) {
			x[tid] = 0;
		}
	}
}

__global__ void uniformToBernoulli(float * x, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		if (x[tid] > 0.5) {
			x[tid] = 1;
		} else {
			x[tid] = 0;
		}
	}
}

__global__ void addRepeated(float * dest, float * source, size_t destLen, size_t sourceLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid % sourceLen];
	}
}

__global__ void addRepeatedPow2(float * dest, float * source, size_t destLen, size_t srcMask) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += source[tid & srcMask];
	}
}

__global__ void addScaler(float s, float * dest, size_t destLen) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < destLen) {
		dest[tid] += s;
	}
}
`

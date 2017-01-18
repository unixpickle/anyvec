//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"

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
}

func newMathKernels() (kernels *mathKernels, err error) {
	ptx := unsafe.Pointer(C.CString(kernelPTX))
	defer C.free(ptx)

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
		cName := C.CString(nameStr)
		defer C.free(unsafe.Pointer(cName))
		var kernel C.CUfunction
		cuRes := C.cuModuleGetFunction(&kernel, module, cName)
		if err := cuError("cuModuleGetFunction", cuRes); err != nil {
			return nil, err
		}
		kernelMap[nameStr] = kernel
	}

	return &mathKernels{
		module:  module,
		kernels: kernelMap,
	}, nil
}

func (m *mathKernels) Destroy() {
	C.cuModuleUnload(m.module)
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

// ScaleRepeated32 scales a target by a repeated vector.
func (m *mathKernels) ScaleRepeated32(dstLen, srcLen int, dst, src unsafe.Pointer) error {
	log2 := uint(math.Log2(float64(srcLen)))
	if (1 << log2) == srcLen {
		return m.call2Asym("scaleRepeatedPow2", dstLen, srcLen-1, dst, src)
	} else {
		return m.call2Asym("scaleRepeated", dstLen, srcLen, dst, src)
	}
}

// AddScaler32 adds a scaler to a target.
func (m *mathKernels) AddScaler32(n int, alpha float32, v unsafe.Pointer) error {
	return m.call1Scaler("addScaler", n, alpha, v)
}

// AddChunks32 adds a set of scalers in chunks.
func (m *mathKernels) AddChunks32(chunkCount, chunkSize int, dst, scal unsafe.Pointer) error {
	return m.call2Asym("addChunks", chunkCount*chunkSize, chunkSize, dst, scal)
}

// Compare32 runs a comparison.
func (m *mathKernels) Compare32(n int, alpha float32, v unsafe.Pointer, c compareType) error {
	switch c {
	case lessThan:
		return m.call1Scaler("lessThan", n, alpha, v)
	case greaterThan:
		return m.call1Scaler("greaterThan", n, alpha, v)
	case equalTo:
		return m.call1Scaler("equalTo", n, alpha, v)
	}
	panic("unknown compare type")
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
	"addRepeated", "addRepeatedPow2", "scaleRepeated", "scaleRepeatedPow2",
	"addScaler", "addChunks", "lessThan", "greaterThan", "equalTo"}

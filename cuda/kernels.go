//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"

const CUjit_option * nullJitOptions = NULL;
const void ** nullPtrPtr = NULL;

void kernel_sizes(int n, unsigned int * block, unsigned int * grid) {
	*block = 128;
	if (n < *block) {
		*block = (unsigned int)n;
		*grid = 1;
	} else {
		*grid = (unsigned int)(n / (*block));
		if (n%(*block) != 0) {
			(*grid)++;
		}
	}
}

CUresult anyvec_cuda_call2(CUfunction f, int n, void * p1, void * p2) {
	void * args[] = {&p1, &p2, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call1(CUfunction f, int n, void * p1) {
	void * args[] = {&p1, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call2_asym(CUfunction f, int n1, int n2, void * p1, void * p2) {
	void * args[] = {&p1, &p2, &n1, &n2};
	unsigned int blockSize, gridSize;
	kernel_sizes(n1, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call1_scaler(CUfunction f, int n, float scaler, void * p1) {
	void * args[] = {&scaler, &p1, &n};
	unsigned int blockSize, gridSize;
	kernel_sizes(n, &blockSize, &gridSize);
	return cuLaunchKernel(f, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
}

CUresult anyvec_cuda_call_addlogs(CUfunction f, int rows, int cols, void * dst,
	void * src, int threadCount) {
	void * args[] = {&dst, &src, &cols};
	unsigned int gridX = ((unsigned int)cols + threadCount - 1) / (unsigned int)(threadCount);
	unsigned int sharedSize = 4 * threadCount;
	return cuLaunchKernel(f, gridX, (unsigned int)rows, 1, threadCount, 1, 1,
		sharedSize, NULL, args, NULL);
}

CUresult anyvec_cuda_call_map(CUfunction f, int n, void * table, void * p1, void * p2) {
	void * args[] = {&p1, &p2, &table, &n};
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
	module    C.CUmodule
	kernels   map[string]C.CUfunction
	allocator allocator
}

func newMathKernels(a allocator) (kernels *mathKernels, err error) {
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
		module:    module,
		kernels:   kernelMap,
		allocator: a,
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

// Log32 performs element-wise logarithms.
func (m *mathKernels) Log32(n int, v unsafe.Pointer) error {
	return m.call1("logElements", n, v)
}

// Tanh32 performs element-wise hyperbolic tangent.
func (m *mathKernels) Tanh32(n int, v unsafe.Pointer) error {
	return m.call1("tanhElements", n, v)
}

// Sin32 performs element-wise sine.
func (m *mathKernels) Sin32(n int, v unsafe.Pointer) error {
	return m.call1("sinElements", n, v)
}

// Sigmoid32 performs element-wise sigmoid.
func (m *mathKernels) Sigmoid32(n int, v unsafe.Pointer) error {
	return m.call1("sigmoidElements", n, v)
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

// SubChunks32 is like AddChunks32, but for subtraction.
func (m *mathKernels) SubChunks32(chunkCount, chunkSize int, dst, scal unsafe.Pointer) error {
	return m.call2Asym("subChunks", chunkCount*chunkSize, chunkSize, dst, scal)
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

// AddLogs32 performs addition in the log domain.
func (m *mathKernels) AddLogs32(rows, cols int, dst, src unsafe.Pointer) error {
	threads := 256

	for threads/2 >= cols && threads > 32 {
		threads /= 2
	}

	k := m.kernels["addLogs"]

	freeSrc := false
	for cols > threads {
		destCols := (cols + threads - 1) / threads
		destSize := destCols * rows * 4
		tempDest, err := m.allocator.Alloc(destSize)
		if err != nil {
			return err
		}
		res := C.anyvec_cuda_call_addlogs(k, C.int(rows), C.int(cols),
			tempDest, src, C.int(threads))
		if freeSrc {
			m.allocator.Free(src)
		}
		if err := cuError("cuLaunchKernel", res); err != nil {
			m.allocator.Free(tempDest)
			return err
		}
		src = tempDest
		cols = destCols
		freeSrc = true
	}

	res := C.anyvec_cuda_call_addlogs(k, C.int(rows), C.int(cols),
		dst, src, C.int(threads))
	if freeSrc {
		m.allocator.Free(src)
	}
	return m.doneKernel(res)
}

// PowScaler32 raises the entries to a power.
func (m *mathKernels) PowScaler32(n int, p float32, v unsafe.Pointer) error {
	return m.call1Scaler("powScaler", n, p, v)
}

// MapForward32 performs forward mapping, like
//
//     dst[i] = src[table[i]]
//
func (m *mathKernels) MapForward32(tableSize int, table, dst, src unsafe.Pointer) error {
	return m.callMap("mapForward", tableSize, table, dst, src)
}

// MapBackward32 performs backward mapping, like
//
//     dst[table[i]] += src[i]
//
func (m *mathKernels) MapBackward32(tableSize int, table, dst, src unsafe.Pointer) error {
	return m.callMap("mapBackward", tableSize, table, dst, src)
}

// MapMax32 assigns entries to a map based on the maximum
// columns in each row.
func (m *mathKernels) MapMax32(rows, cols int, table, v unsafe.Pointer) error {
	return m.call2Asym("mapMax", rows, cols, table, v)
}

func (m *mathKernels) call1(name string, n int, v unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call1(k, C.int(n), v))
}

func (m *mathKernels) call1Scaler(name string, n int, s float32, v unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call1_scaler(k, C.int(n), C.float(s), v))
}

func (m *mathKernels) call2(name string, n int, v1, v2 unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call2(k, C.int(n), v1, v2))
}

func (m *mathKernels) call2Asym(name string, n1, n2 int, v1, v2 unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call2_asym(k, C.int(n1), C.int(n2), v1, v2))
}

func (m *mathKernels) callMap(name string, tableSize int, table, dst, src unsafe.Pointer) error {
	k := m.kernels[name]
	return m.doneKernel(C.anyvec_cuda_call_map(k, C.int(tableSize), table, dst, src))
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

var mathKernelNames = []string{"divElements", "expElements", "logElements", "tanhElements",
	"sinElements", "sigmoidElements", "clipPositive", "shiftRandUniform", "uniformToBernoulli",
	"addRepeated", "addRepeatedPow2", "scaleRepeated", "scaleRepeatedPow2",
	"addScaler", "addChunks", "subChunks", "lessThan", "greaterThan", "equalTo",
	"addLogs", "powScaler", "mapForward", "mapBackward", "mapMax"}

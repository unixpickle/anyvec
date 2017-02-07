package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

extern CUresult cuSuccess;
extern cublasStatus_t cublasSuccess;
extern cublasSideMode_t sideModeRight;
extern cublasSideMode_t sideModeLeft;

CUresult anyvec_cuda_set1_32(size_t n, void * vec) {
	float datum = 1;
	unsigned int datumRaw = *((unsigned int *)&datum);
	return cuMemsetD32((CUdeviceptr)vec, datumRaw, n);
}
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/unixpickle/anyvec"
)

// ops32 provides a small wrapper around various vector
// operations (e.g. cuBLAS calls or kernel invocations).
type ops32 struct {
	h *Handle
}

// Scal scales the device vector x by the scaler s.
// The vector contains n elements.
func (o ops32) Scal(n int, s float32, x *buffer) {
	o.h.loop.RunCUBLASAsync(func(blas C.cublasHandle_t) {
		mustBLAS("Sscal", C.cublasSscal(blas, C.int(n), (*C.float)(&s),
			(*C.float)(x.ptr), 1))
		runtime.KeepAlive(x)
	})
}

// Dot takes the dot product of two device vectors.
// The vectors both contain n elements.
func (o ops32) Dot(n int, x, y *buffer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Sdot", C.cublasSdot(blas, C.int(n), (*C.float)(x.ptr), 1,
			(*C.float)(y.ptr), 1, &tempRes))
		runtime.KeepAlive(x)
		runtime.KeepAlive(y)
		res = float32(tempRes)
	})
	return res
}

// Axpy computes alpha*x + y and stores the result in y.
// The vectors both contain n elements.
func (o ops32) Axpy(n int, alpha float32, x, y *buffer) {
	o.h.loop.RunCUBLASAsync(func(blas C.cublasHandle_t) {
		a := C.float(alpha)
		mustBLAS("Saxpy", C.cublasSaxpy(blas, C.int(n), (*C.float)(&a),
			(*C.float)(x.ptr), 1, (*C.float)(y.ptr), 1))
		runtime.KeepAlive(x)
		runtime.KeepAlive(y)
	})
}

// Gemm performs the classic BLAS gemm operation.
func (o ops32) Gemm(transA, transB bool, m, n, k int, alpha float32, a *buffer,
	lda int, b *buffer, ldb int, beta float32, c *buffer, ldc int) {
	o.h.loop.RunCUBLASAsync(func(blas C.cublasHandle_t) {
		alphaC := C.float(alpha)
		betaC := C.float(beta)
		// Stuff is ordered to emulate column-major storage.
		mustBLAS("Sgemm", C.cublasSgemm(blas, blasTransposeOp(transB),
			blasTransposeOp(transA), C.int(n), C.int(m), C.int(k),
			(*C.float)(&alphaC), (*C.float)(b.ptr), C.int(ldb),
			(*C.float)(a.ptr), C.int(lda), (*C.float)(&betaC),
			(*C.float)(c.ptr), C.int(ldc)))
		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
		runtime.KeepAlive(c)
	})
}

// Asum computes the 1-norm of the vector.
// The vector contains n elements.
func (o ops32) Asum(n int, v *buffer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Sasum", C.cublasSasum(blas, C.int(n), (*C.float)(v.ptr), 1, &tempRes))
		runtime.KeepAlive(v)
		res = float32(tempRes)
	})
	return res
}

// Amax computes the absolute value of the element with
// the largest absolute value in the vector.
// The vector contains n elements.
func (o ops32) Amax(n int, v *buffer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var resIdx C.int
		mustBLAS("Isamax", C.cublasIsamax(blas, C.int(n), (*C.float)(v.ptr), 1, &resIdx))
		var tempRes C.float

		// Isamax uses 1-based indexing for Fortran compatibility.
		offPtr := unsafe.Pointer(uintptr(v.ptr) + 4*uintptr(resIdx-1))

		must(cudaError("cudaMemcpy", C.cudaMemcpy(unsafe.Pointer(&tempRes), offPtr,
			4, C.cudaMemcpyDeviceToHost)))
		runtime.KeepAlive(v)
		res = float32(math.Abs(float64(tempRes)))
	})
	return res
}

// Nrm2 computes the Euclidean norm.
// The vector contains n elements.
func (o ops32) Nrm2(n int, v *buffer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Snrm2", C.cublasSnrm2(blas, C.int(n), (*C.float)(v.ptr), 1, &tempRes))
		runtime.KeepAlive(v)
		res = float32(tempRes)
	})
	return res
}

// Mul performs component-wise multiplication and stores
// the result in a.
// The vectors both contain n elements.
func (o ops32) Mul(n int, a, b *buffer) {
	o.h.loop.RunCUBLASAsync(func(blas C.cublasHandle_t) {
		mustBLAS("Sdgmm", C.cublasSdgmm(blas, C.sideModeRight, 1, C.int(n),
			(*C.float)(a.ptr), 1, (*C.float)(b.ptr), 1, (*C.float)(a.ptr), 1))
		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
	})
}

// MulChunks multiplies chunks of vec by scalers from
// scales.
// vec contains chunkCount chunks, each of chunkSize
// components.
// scales contains one component per chunk.
func (o ops32) MulChunks(chunkCount, chunkSize int, vec, scales *buffer) {
	o.h.loop.RunCUBLASAsync(func(blas C.cublasHandle_t) {
		mustBLAS("Sdgmm", C.cublasSdgmm(blas, C.sideModeRight, C.int(chunkSize),
			C.int(chunkCount), (*C.float)(vec.ptr), C.int(chunkSize),
			(*C.float)(scales.ptr), 1, (*C.float)(vec.ptr), C.int(chunkSize)))
		runtime.KeepAlive(vec)
		runtime.KeepAlive(scales)
	})
}

// AddChunks is like ScaleChunks, but for addition.
func (o ops32) AddChunks(chunkCount, chunkSize int, vec, scales *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.AddChunks32(chunkCount, chunkSize, vec.ptr, scales.ptr))
		runtime.KeepAlive(vec)
		runtime.KeepAlive(scales)
	})
}

// Sum computes the sum of the components in the vector.
// The vector contains n elements.
func (o ops32) Sum(n int, a *buffer) float32 {
	if n == 0 {
		return 0
	}
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		tempBuf, err := o.h.allocator.Alloc(n * 4)
		must(err)
		defer o.h.allocator.Free(tempBuf)
		must(cuError("cuMemsetD32", C.anyvec_cuda_set1_32(C.size_t(n), tempBuf)))
		var tempRes C.float
		mustBLAS("Sdot", C.cublasSdot(blas, C.int(n), (*C.float)(a.ptr), 1,
			(*C.float)(tempBuf), 1, &tempRes))
		runtime.KeepAlive(a)
		res = float32(tempRes)
	})
	return res
}

// Div divides the components of a by those of b and
// stores the result in a.
// The vectors both contain n elements.
func (o ops32) Div(n int, a, b *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Div32(n, a.ptr, b.ptr))
		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
	})
}

// ElemMax sets the elements of a to be the component-wise
// maximum between a and b.
// The vectors both contain n elements.
func (o ops32) ElemMax(n int, a, b *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.ElemMax32(n, a.ptr, b.ptr))
		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
	})
}

// Exp exponentiates the vector components.
// The vector contains n elements.
func (o ops32) Exp(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Exp32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// Log takes the natural log of the vector components.
// The vector contains n elements.
func (o ops32) Log(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Log32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// Tanh computes the hyperbolic tangent of the vector
// components.
// The vector contains n elements.
func (o ops32) Tanh(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Tanh32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// Sin computes the sine of the vector components.
// The vector contains n elements.
func (o ops32) Sin(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Sin32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// Sigmoid computes the logistic sigmoid of the vector
// components.
// The vector contains n elements.
func (o ops32) Sigmoid(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Sigmoid32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// ClipPos takes MAX(0, x) for every component x.
// The vector contains n elements.
func (o ops32) ClipPos(n int, a *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.ClipPos32(n, a.ptr))
		runtime.KeepAlive(a)
	})
}

// GenRand randomizes the contents of a vector.
// The vector contains n elements.
func (o ops32) GenRand(n int, a *buffer, dist anyvec.ProbDist) {
	o.h.runWithRandAsync(func() {
		switch dist {
		case anyvec.Bernoulli:
			must(o.h.rand.Bernoulli32(o.h.kernels, n, a.ptr))
		case anyvec.Uniform:
			must(o.h.rand.Uniform32(o.h.kernels, n, a.ptr))
		case anyvec.Normal:
			must(o.h.rand.Norm32(n, a.ptr))
		default:
			panic(fmt.Sprintf("unsupported distribution: %v", dist))
		}
		runtime.KeepAlive(a)
	})
}

// AddRepeated updates the components of dst as
// dst[i] += src[i%srcLen].
//
// The dst vector contains dstLen components.
// The src vector contains srcLen components.
func (o ops32) AddRepeated(dstLen, srcLen int, dst, src *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.AddRepeated32(dstLen, srcLen, dst.ptr, src.ptr))
		runtime.KeepAlive(dst)
		runtime.KeepAlive(src)
	})
}

// ScaleRepeated updates the components of dst as
// dst[i] *= src[i%srcLen].
//
// The dst vector contains dstLen components.
// The src vector contains srcLen components.
func (o ops32) ScaleRepeated(dstLen, srcLen int, dst, src *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.ScaleRepeated32(dstLen, srcLen, dst.ptr, src.ptr))
		runtime.KeepAlive(dst)
		runtime.KeepAlive(src)
	})
}

// AddScaler adds a scaler s to the components of x.
// The vector contains n elements.
func (o ops32) AddScaler(n int, alpha float32, v *buffer) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.AddScaler32(n, alpha, v.ptr))
		runtime.KeepAlive(v)
	})
}

// Compare performs a comparison, assigning elements to 1
// if the comparison is true.
// The vector contains n elements.
func (o ops32) Compare(n int, alpha float32, v *buffer, c compareType) {
	o.h.runWithKernelsAsync(func() {
		must(o.h.kernels.Compare32(n, alpha, v.ptr, c))
		runtime.KeepAlive(v)
	})
}

// AddLogs performs addition in the log domain.
func (o ops32) AddLogs(rows, cols int, src *buffer) *buffer {
	var res *buffer
	o.h.runWithKernels(func() {
		resPtr, err := o.h.allocator.Alloc(4 * rows)
		must(err)
		if err := o.h.kernels.AddLogs32(rows, cols, resPtr, src.ptr); err != nil {
			o.h.allocator.Free(resPtr)
			panic(err)
		}
		runtime.KeepAlive(src)
		res = newBufferPtr(o.h, 4*rows, resPtr, true)
	})
	return res
}

// LogSoftmax computes the log of the softmax.
func (o ops32) LogSoftmax(rows, cols int, vecs *buffer) {
	o.h.runWithKernelsAsync(func() {
		sums, err := o.h.allocator.Alloc(4 * rows)
		must(err)
		defer o.h.allocator.Free(sums)
		must(o.h.kernels.AddLogs32(rows, cols, sums, vecs.ptr))
		must(o.h.kernels.SubChunks32(rows, cols, vecs.ptr, sums))
		runtime.KeepAlive(vecs)
	})
}

// Pow raises the components to the given power.
func (o ops32) Pow(n int, p float32, v *buffer) {
	o.h.runWithKernelsAsync(func() {
		o.h.kernels.PowScaler32(n, p, v.ptr)
		runtime.KeepAlive(v)
	})
}

// MapMax creates a map for the maximum entry in each row.
func (o ops32) MapMax(rows, cols int, src *buffer) *buffer {
	var res *buffer
	o.h.runWithKernels(func() {
		table, err := o.h.allocator.Alloc(rows * 4)
		if err != nil {
			panic(err)
		}
		err = o.h.kernels.MapMax32(rows, cols, table, src.ptr)
		runtime.KeepAlive(src)
		if err != nil {
			o.h.allocator.Free(table)
			panic(err)
		}
		res = newBufferPtr(o.h, rows*4, table, true)
	})
	return res
}

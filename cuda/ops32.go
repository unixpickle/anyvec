//+build !nocuda

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
func (o ops32) Scal(n int, s float32, x unsafe.Pointer) {
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		mustBLAS("Sscal", C.cublasSscal(blas, C.int(n), (*C.float)(&s),
			(*C.float)(x), 1))
	})
}

// Dot takes the dot product of two device vectors.
// The vectors both contain n elements.
func (o ops32) Dot(n int, x, y unsafe.Pointer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Sdot", C.cublasSdot(blas, C.int(n), (*C.float)(x), 1,
			(*C.float)(y), 1, &tempRes))
		res = float32(tempRes)
	})
	return res
}

// Axpy computes alpha*x + y and stores the result in y.
// The vectors both contain n elements.
func (o ops32) Axpy(n int, alpha float32, x, y unsafe.Pointer) {
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		a := C.float(alpha)
		mustBLAS("Saxpy", C.cublasSaxpy(blas, C.int(n), (*C.float)(&a),
			(*C.float)(x), 1, (*C.float)(y), 1))
	})
}

// Gemm performs the classic BLAS gemm operation.
func (o ops32) Gemm(transA, transB bool, m, n, k int, alpha float32, a unsafe.Pointer,
	lda int, b unsafe.Pointer, ldb int, beta float32, c unsafe.Pointer, ldc int) {
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		alphaC := C.float(alpha)
		betaC := C.float(beta)
		// Stuff is ordered to emulate column-major storage.
		mustBLAS("Sgemm", C.cublasSgemm(blas, blasTransposeOp(transB),
			blasTransposeOp(transA), C.int(n), C.int(m), C.int(k),
			(*C.float)(&alphaC), (*C.float)(b), C.int(ldb),
			(*C.float)(a), C.int(lda), (*C.float)(&betaC),
			(*C.float)(c), C.int(ldc)))
	})
}

// Asum computes the 1-norm of the vector.
// The vector contains n elements.
func (o ops32) Asum(n int, v unsafe.Pointer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Sasum", C.cublasSasum(blas, C.int(n), (*C.float)(v), 1, &tempRes))
		res = float32(tempRes)
	})
	return res
}

// Amax computes the absolute value of the element with
// the largest absolute value in the vector.
// The vector contains n elements.
func (o ops32) Amax(n int, v unsafe.Pointer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var resIdx C.int
		mustBLAS("Isamax", C.cublasIsamax(blas, C.int(n), (*C.float)(v), 1, &resIdx))
		var tempRes C.float

		// Isamax uses 1-based indexing for Fortran compatibility.
		offPtr := unsafe.Pointer(uintptr(v) + 4*uintptr(resIdx-1))

		must(cudaError("cudaMemcpy", C.cudaMemcpy(unsafe.Pointer(&tempRes), offPtr,
			4, C.cudaMemcpyDeviceToHost)))
		res = float32(math.Abs(float64(tempRes)))
	})
	return res
}

// Nrm2 computes the Euclidean norm.
// The vector contains n elements.
func (o ops32) Nrm2(n int, v unsafe.Pointer) float32 {
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		mustBLAS("Snrm2", C.cublasSnrm2(blas, C.int(n), (*C.float)(v), 1, &tempRes))
		res = float32(tempRes)
	})
	return res
}

// Mul performs component-wise multiplication and stores
// the result in a.
// The vectors both contain n elements.
func (o ops32) Mul(n int, a, b unsafe.Pointer) {
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		mustBLAS("Sdgmm", C.cublasSdgmm(blas, C.sideModeRight, 1, C.int(n),
			(*C.float)(a), 1, (*C.float)(b), 1, (*C.float)(a), 1))
	})
}

// MulChunks multiplies chunks of vec by scalers from
// scales.
// vec contains chunkCount chunks, each of chunkSize
// components.
// scales contains one component per chunk.
func (o ops32) MulChunks(chunkCount, chunkSize int, vec, scales unsafe.Pointer) {
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		mustBLAS("Sdgmm", C.cublasSdgmm(blas, C.sideModeRight, C.int(chunkSize),
			C.int(chunkCount), (*C.float)(vec), C.int(chunkSize), (*C.float)(scales), 1,
			(*C.float)(vec), C.int(chunkSize)))
	})
}

// AddChunks is like ScaleChunks, but for addition.
func (o ops32) AddChunks(chunkCount, chunkSize int, vec, scales unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.AddChunks32(chunkCount, chunkSize, vec, scales))
	})
}

// Sum computes the sum of the components in the vector.
// The vector contains n elements.
func (o ops32) Sum(n int, a unsafe.Pointer) float32 {
	if n == 0 {
		return 0
	}
	var res float32
	o.h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempBuf unsafe.Pointer
		must(cudaError("cudaMalloc", C.cudaMalloc(&tempBuf, C.size_t(n*4))))
		defer C.cudaFree(tempBuf)
		must(cuError("cuMemsetD32", C.anyvec_cuda_set1_32(C.size_t(n), tempBuf)))
		var tempRes C.float
		mustBLAS("Sdot", C.cublasSdot(blas, C.int(n), (*C.float)(a), 1,
			(*C.float)(tempBuf), 1, &tempRes))
		res = float32(tempRes)
	})
	return res
}

// Div divides the components of a by those of b and
// stores the result in a.
// The vectors both contain n elements.
func (o ops32) Div(n int, a, b unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.Div32(n, a, b))
	})
}

// Exp exponentiates the vector components.
// The vector contains n elements.
func (o ops32) Exp(n int, a unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.Exp32(n, a))
	})
}

// Tanh computes the hyperbolic tangent of the vector
// components.
// The vector contains n elements.
func (o ops32) Tanh(n int, a unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.Tanh32(n, a))
	})
}

// Sin computes the sine of the vector components.
// The vector contains n elements.
func (o ops32) Sin(n int, a unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.Sin32(n, a))
	})
}

// ClipPos takes MAX(0, x) for every component x.
// The vector contains n elements.
func (o ops32) ClipPos(n int, a unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.ClipPos32(n, a))
	})
}

// GenRand randomizes the contents of a vector.
// The vector contains n elements.
func (o ops32) GenRand(n int, a unsafe.Pointer, dist anyvec.ProbDist) {
	o.h.runWithRand(func() {
		switch dist {
		case anyvec.Bernoulli:
			must(o.h.rand.Bernoulli32(o.h.kernels, n, a))
		case anyvec.Uniform:
			must(o.h.rand.Uniform32(o.h.kernels, n, a))
		case anyvec.Normal:
			must(o.h.rand.Norm32(n, a))
		default:
			panic(fmt.Sprintf("unsupported distribution: %v", dist))
		}
	})
}

// AddRepeated updates the components of dst as
// dst[i] += src[i%srcLen].
//
// The dst vector contains dstLen components.
// The src vector contains srcLen components.
func (o ops32) AddRepeated(dstLen, srcLen int, dst, src unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.AddRepeated32(dstLen, srcLen, dst, src))
	})
}

// ScaleRepeated updates the components of dst as
// dst[i] *= src[i%srcLen].
//
// The dst vector contains dstLen components.
// The src vector contains srcLen components.
func (o ops32) ScaleRepeated(dstLen, srcLen int, dst, src unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.ScaleRepeated32(dstLen, srcLen, dst, src))
	})
}

// AddScaler adds a scaler s to the components of x.
// The vector contains n elements.
func (o ops32) AddScaler(n int, alpha float32, v unsafe.Pointer) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.AddScaler32(n, alpha, v))
	})
}

// Compare performs a comparison, assigning elements to 1
// if the comparison is true.
// The vector contains n elements.
func (o ops32) Compare(n int, alpha float32, v unsafe.Pointer, c compareType) {
	o.h.runWithKernels(func() {
		must(o.h.kernels.Compare32(n, alpha, v, c))
	})
}

// AddLogs performs addition in the log domain.
func (o ops32) AddLogs(rows, cols int, src unsafe.Pointer) unsafe.Pointer {
	var res unsafe.Pointer
	o.h.runWithKernels(func() {
		must(cudaError("cudaMalloc", C.cudaMalloc(&res, C.size_t(4*rows))))
		if err := o.h.kernels.AddLogs32(rows, cols, res, src); err != nil {
			C.cudaFree(res)
			panic(err)
		}
	})
	return res
}

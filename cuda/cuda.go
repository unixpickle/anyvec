//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

const cublasOperation_t noTranspose = CUBLAS_OP_N;
const cublasOperation_t transpose = CUBLAS_OP_T;

cublasHandle_t anyvec_cuda_new_handle() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUDA_SUCCESS) {
        return NULL;
    }
    return handle;
}

void anyvec_cuda_destroy_handle(cublasHandle_t handle) {
    cublasDestroy(handle);
}

int anyvec_cuda_is_null(void * ptr) {
    return ptr == NULL;
}

int anyvec_cuda_is_success(cublasStatus_t s) {
	return s == CUBLAS_STATUS_SUCCESS;
}

void * anyvec_cuda_alloc(size_t size) {
	void * ptr;
	if (cudaMalloc(&ptr, size) != cudaSuccess) {
		return NULL;
	}
	return ptr;
}
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// These errors indicate various CUDA-related failures.
var (
	ErrHandleCreation = errors.New("failed to create cuBLAS handle")
	ErrMemoryAlloc    = errors.New("failed to allocate CUDA memory")
	ErrMemoryZero     = errors.New("failed to zero CUDA memory")
	ErrMemorySet      = errors.New("failed to set CUDA memory")
	ErrMemoryCopy     = errors.New("failed to transfer CUDA memory")
	ErrMatrixMultiply = errors.New("matrix multiplication failed")
)

// A Handle manages an internal CUDA context.
type Handle struct {
	handlePtr C.cublasHandle_t
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	handle := C.anyvec_cuda_new_handle()
	if C.anyvec_cuda_is_null(unsafe.Pointer(handle)) != C.int(0) {
		return nil, ErrHandleCreation
	}
	res := &Handle{handlePtr: handle}
	runtime.SetFinalizer(res, func(obj *Handle) {
		C.anyvec_cuda_destroy_handle(obj.handlePtr)
	})
	return res, nil
}

func (h *Handle) sscal(n int, s float32, x unsafe.Pointer) {
	h.panicOnErr(C.cublasSscal(h.handlePtr, C.int(n), (*C.float)(&s), (*C.float)(x), 1))
}

func (h *Handle) sdot(n int, x, y unsafe.Pointer) float32 {
	var res C.float
	h.panicOnErr(C.cublasSdot(h.handlePtr, C.int(n), (*C.float)(x), 1, (*C.float)(y),
		1, &res))
	return float32(res)
}

func (h *Handle) saxpy(n int, alpha float32, x, y unsafe.Pointer) {
	a := C.float(alpha)
	h.panicOnErr(C.cublasSaxpy(h.handlePtr, C.int(n), (*C.float)(&a), (*C.float)(x),
		1, (*C.float)(y), 1))
}

func (h *Handle) sgemm(transA, transB bool, m, n, k int, alpha float32, a unsafe.Pointer,
	lda int, b unsafe.Pointer, ldb int, beta float32, c unsafe.Pointer, ldc int) {
	alphaC := C.float(alpha)
	betaC := C.float(beta)
	// Stuff is moved around to deal with column-major order.
	h.panicOnErr(C.cublasSgemm(h.handlePtr, h.transposeOp(transB),
		h.transposeOp(transA), C.int(n), C.int(m), C.int(k),
		(*C.float)(&alphaC), (*C.float)(b), C.int(ldb),
		(*C.float)(a), C.int(lda), (*C.float)(&betaC),
		(*C.float)(c), C.int(ldc)))
}

func (h *Handle) panicOnErr(s C.cublasStatus_t) {
	if C.anyvec_cuda_is_success(s) == C.int(0) {
		panic("cuBLAS operation failed")
	}
}

func (h *Handle) transposeOp(trans bool) C.cublasOperation_t {
	if trans {
		return C.transpose
	}
	return C.noTranspose
}

// A buffer is an on-device memory buffer.
type buffer struct {
	size int
	ptr  unsafe.Pointer
}

// newBuffer allocates a buffer.
func newBuffer(size int) (*buffer, error) {
	buff := C.anyvec_cuda_alloc(C.size_t(size))
	if C.anyvec_cuda_is_null(buff) != C.int(0) {
		return nil, ErrMemoryAlloc
	}
	res := &buffer{
		size: size,
		ptr:  buff,
	}
	runtime.SetFinalizer(res, func(b *buffer) {
		C.cudaFree(b.ptr)
	})
	return res, nil
}

// Len returns the buffer's length in bytes.
func (b *buffer) Len() int {
	return b.size
}

// Clear zeroes the buffer.
func (b *buffer) Clear() error {
	res := C.cudaMemset(b.ptr, 0, C.size_t(b.size))
	runtime.KeepAlive(b)
	if res != C.cudaSuccess {
		return ErrMemoryZero
	}
	return nil
}

// Set copies the contents of a buffer into b.
func (b *buffer) Set(b1 *buffer) error {
	if b1.size != b.size {
		return errors.New("buffer sizes do not match")
	}
	res := b.memcpyErr(C.cudaMemcpy(b.ptr, b1.ptr, C.size_t(b.size),
		C.cudaMemcpyDeviceToDevice))
	runtime.KeepAlive(b1)
	runtime.KeepAlive(b)
	return res
}

// Set32 copies 32-bit floats into the buffer.
func (b *buffer) Set32(source []float32) error {
	if len(source)*4 > b.size {
		panic("buffer overflow")
	}
	res := b.memcpyErr(C.cudaMemcpy(b.ptr, unsafe.Pointer(&source[0]),
		C.size_t(len(source)*4), C.cudaMemcpyHostToDevice))
	runtime.KeepAlive(source)
	runtime.KeepAlive(b)
	return res
}

// SetRepeated32 copies the same 32-bits again and again
// to fill the buffer.
func (b *buffer) SetRepeated32(v float32) error {
	if b.size%4 != 0 {
		panic("size not divisible by 4")
	}
	buf := make([]float32, b.size/4)
	for i := range buf {
		buf[i] = v
	}
	return b.Set32(buf)
}

// Set64 copies 64-bit floats into the buffer.
func (b *buffer) Set64(source []float64) error {
	if len(source)*8 > b.size {
		panic("buffer overflow")
	}
	res := b.memcpyErr(C.cudaMemcpy(b.ptr, unsafe.Pointer(&source[0]),
		C.size_t(len(source)*8), C.cudaMemcpyHostToDevice))
	runtime.KeepAlive(source)
	runtime.KeepAlive(b)
	return res
}

// SetRepeated64 copies the same 64-bits again and again
// to fill the buffer.
func (b *buffer) SetRepeated64(v float64) error {
	if b.size%8 != 0 {
		panic("size not divisible by 8")
	}
	buf := make([]float64, b.size/8)
	for i := range buf {
		buf[i] = v
	}
	return b.Set64(buf)
}

// Get32 copies 32-bit floats out of the buffer.
func (b *buffer) Get32(source []float32) error {
	if len(source)*4 > b.size {
		panic("buffer overflow")
	}
	res := b.memcpyErr(C.cudaMemcpy(unsafe.Pointer(&source[0]), b.ptr,
		C.size_t(len(source)*4), C.cudaMemcpyDeviceToHost))
	runtime.KeepAlive(source)
	runtime.KeepAlive(b)
	return res
}

// Get64 copies 64-bit floats out of the buffer.
func (b *buffer) Get64(source []float64) error {
	if len(source)*8 > b.size {
		panic("buffer overflow")
	}
	res := b.memcpyErr(C.cudaMemcpy(unsafe.Pointer(&source[0]), b.ptr,
		C.size_t(len(source)*8), C.cudaMemcpyDeviceToHost))
	runtime.KeepAlive(source)
	runtime.KeepAlive(b)
	return res
}

func (b *buffer) memcpyErr(status C.cudaError_t) error {
	if status == C.cudaSuccess {
		return nil
	}
	return ErrMemoryCopy
}

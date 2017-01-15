//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

const cublasOperation_t noTranspose = CUBLAS_OP_N;
const cublasOperation_t transpose = CUBLAS_OP_T;
const cublasSideMode_t sideMode = CUBLAS_SIDE_LEFT;
CUresult cuSuccess = CUDA_SUCCESS;

int anyvec_cuda_is_success(cublasStatus_t s) {
	return s == CUBLAS_STATUS_SUCCESS;
}

int anyvec_cuda_is_null(void * ptr) {
    return ptr == NULL;
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
	ErrMakeHandle     = errors.New("failed to create cuBLAS handle")
	ErrGetDevice      = errors.New("failed to get CUDA device")
	ErrMakeContext    = errors.New("failed to create CUDA context")
	ErrMemoryAlloc    = errors.New("failed to allocate CUDA memory")
	ErrMemoryZero     = errors.New("failed to zero CUDA memory")
	ErrMemorySet      = errors.New("failed to set CUDA memory")
	ErrMemoryCopy     = errors.New("failed to transfer CUDA memory")
	ErrMatrixMultiply = errors.New("matrix multiplication failed")
)

// A Handle manages an internal CUDA context.
type Handle struct {
	loop    *cudaLoop
	kernels *mathKernels
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	err := createMainLoop()
	if err != nil {
		return nil, err
	}
	res := &Handle{loop: getMainLoop()}
	runtime.SetFinalizer(res, func(obj *Handle) {
		obj.loop.Run(func() {
			if obj.kernels != nil {
				obj.kernels.Destroy()
			}
		})
	})
	return res, nil
}

func (h *Handle) sscal(n int, s float32, x unsafe.Pointer) {
	h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		h.panicOnErr(C.cublasSscal(blas, C.int(n), (*C.float)(&s), (*C.float)(x), 1))
	})
}

func (h *Handle) sdot(n int, x, y unsafe.Pointer) float32 {
	var res float32
	h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		var tempRes C.float
		h.panicOnErr(C.cublasSdot(blas, C.int(n), (*C.float)(x), 1, (*C.float)(y),
			1, &tempRes))
		res = float32(tempRes)
	})
	return res
}

func (h *Handle) saxpy(n int, alpha float32, x, y unsafe.Pointer) {
	h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		a := C.float(alpha)
		h.panicOnErr(C.cublasSaxpy(blas, C.int(n), (*C.float)(&a), (*C.float)(x),
			1, (*C.float)(y), 1))
	})
}

func (h *Handle) sgemm(transA, transB bool, m, n, k int, alpha float32, a unsafe.Pointer,
	lda int, b unsafe.Pointer, ldb int, beta float32, c unsafe.Pointer, ldc int) {
	h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		alphaC := C.float(alpha)
		betaC := C.float(beta)
		// Stuff is ordered to emulate column-major storage.
		h.panicOnErr(C.cublasSgemm(blas, h.transposeOp(transB),
			h.transposeOp(transA), C.int(n), C.int(m), C.int(k),
			(*C.float)(&alphaC), (*C.float)(b), C.int(ldb),
			(*C.float)(a), C.int(lda), (*C.float)(&betaC),
			(*C.float)(c), C.int(ldc)))
	})
}

func (h *Handle) mul(n int, a, b unsafe.Pointer) {
	h.loop.RunCUBLAS(func(blas C.cublasHandle_t) {
		h.panicOnErr(C.cublasSdgmm(blas, C.sideMode, C.int(n), 1,
			(*C.float)(a), C.int(n), (*C.float)(b), 1, (*C.float)(a), C.int(n)))
	})
}

func (h *Handle) div(n int, a, b unsafe.Pointer) {
	h.runWithKernels(func() error {
		return h.kernels.Div32(a, b, n)
	})
}

func (h *Handle) exp(n int, a unsafe.Pointer) {
	h.runWithKernels(func() error {
		return h.kernels.Exp32(a, n)
	})
}

func (h *Handle) tanh(n int, a unsafe.Pointer) {
	h.runWithKernels(func() error {
		return h.kernels.Tanh32(a, n)
	})
}

func (h *Handle) sin(n int, a unsafe.Pointer) {
	h.runWithKernels(func() error {
		return h.kernels.Sin32(a, n)
	})
}

func (h *Handle) clipPos(n int, a unsafe.Pointer) {
	h.runWithKernels(func() error {
		return h.kernels.ClipPos32(a, n)
	})
}

func (h *Handle) runWithKernels(f func() error) {
	h.loop.Run(func() {
		var err error
		if h.kernels == nil {
			h.kernels, err = newMathKernels()
			if err != nil {
				panic(err)
			}
		}
		err = f()
		if err != nil {
			panic(err)
		}
	})
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
	handle *Handle
	size   int
	ptr    unsafe.Pointer
}

// newBuffer allocates a buffer.
func newBuffer(h *Handle, size int) (*buffer, error) {
	var res *buffer
	var err error
	h.loop.Run(func() {
		buff := C.anyvec_cuda_alloc(C.size_t(size))
		if C.anyvec_cuda_is_null(buff) != C.int(0) {
			err = ErrMemoryAlloc
			return
		}
		res = &buffer{
			handle: h,
			size:   size,
			ptr:    buff,
		}
	})
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(b *buffer) {
		b.handle.loop.Run(func() {
			C.cudaFree(b.ptr)
		})
	})
	return res, nil
}

// newBufferConcat concatenates buffers.
func newBufferConcat(h *Handle, bufs []*buffer) (*buffer, error) {
	var size int
	for _, x := range bufs {
		size += x.size
	}
	buf, err := newBuffer(h, size)
	if err != nil {
		return nil, err
	}
	h.loop.Run(func() {
		var idx uintptr
		for _, x := range bufs {
			dest := unsafe.Pointer(uintptr(buf.ptr) + idx)
			idx += uintptr(x.size)
			err = buf.memcpyErr(C.cudaMemcpy(dest, x.ptr, C.size_t(x.size),
				C.cudaMemcpyDeviceToDevice))
			if err != nil {
				return
			}
		}
	})
	if err != nil {
		return nil, err
	}
	return buf, nil
}

// Len returns the buffer's length in bytes.
func (b *buffer) Len() int {
	return b.size
}

// Clear zeroes the buffer.
func (b *buffer) Clear() error {
	var res C.cudaError_t
	b.handle.loop.Run(func() {
		res = C.cudaMemset(b.ptr, 0, C.size_t(b.size))
	})
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
	var res error
	b.handle.loop.Run(func() {
		res = b.memcpyErr(C.cudaMemcpy(b.ptr, b1.ptr, C.size_t(b.size),
			C.cudaMemcpyDeviceToDevice))
	})
	runtime.KeepAlive(b1)
	runtime.KeepAlive(b)
	return res
}

// Set32 copies 32-bit floats into the buffer.
func (b *buffer) Set32(source []float32) error {
	if len(source)*4 > b.size {
		panic("buffer overflow")
	}
	var res error
	b.handle.loop.Run(func() {
		res = b.memcpyErr(C.cudaMemcpy(b.ptr, unsafe.Pointer(&source[0]),
			C.size_t(len(source)*4), C.cudaMemcpyHostToDevice))
	})
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
	var res error
	b.handle.loop.Run(func() {
		res = b.memcpyErr(C.cudaMemcpy(b.ptr, unsafe.Pointer(&source[0]),
			C.size_t(len(source)*8), C.cudaMemcpyHostToDevice))
	})
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
	var res error
	b.handle.loop.Run(func() {
		res = b.memcpyErr(C.cudaMemcpy(unsafe.Pointer(&source[0]), b.ptr,
			C.size_t(len(source)*4), C.cudaMemcpyDeviceToHost))
	})
	runtime.KeepAlive(source)
	runtime.KeepAlive(b)
	return res
}

// Get64 copies 64-bit floats out of the buffer.
func (b *buffer) Get64(source []float64) error {
	if len(source)*8 > b.size {
		panic("buffer overflow")
	}
	var res error
	b.handle.loop.Run(func() {
		res = b.memcpyErr(C.cudaMemcpy(unsafe.Pointer(&source[0]), b.ptr,
			C.size_t(len(source)*8), C.cudaMemcpyDeviceToHost))
	})
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

//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

const cublasOperation_t noTranspose = CUBLAS_OP_N;
const cublasOperation_t transpose = CUBLAS_OP_T;
const cublasSideMode_t sideMode = CUBLAS_SIDE_LEFT;
const CUresult cuSuccess = CUDA_SUCCESS;

cublasHandle_t anyvec_cuda_new_handle() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUDA_SUCCESS) {
        return NULL;
    }
    return handle;
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
	"sync"
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
	loop *cudaLoop
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	err := createMainLoop()
	if err != nil {
		return nil, err
	}
	res := &Handle{loop: getMainLoop()}
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

var mainLoop *cudaLoop
var mainLoopLock sync.RWMutex

func createMainLoop() error {
	mainLoopLock.Lock()
	defer mainLoopLock.Unlock()
	if mainLoop != nil {
		return nil
	}
	var err error
	mainLoop, err = newCudaLoop()
	return err
}

func getMainLoop() *cudaLoop {
	mainLoopLock.RLock()
	res := mainLoop
	mainLoopLock.RUnlock()
	return res
}

// A cudaLoop runs functions in a dedicated CUDA thread.
type cudaLoop struct {
	ch chan<- *cudaLoopMsg
}

// newCudaLoop creates a new cudaLoop and all of the
// resources associated with it.
func newCudaLoop() (*cudaLoop, error) {
	ch := make(chan *cudaLoopMsg, 1)
	res := &cudaLoop{ch: ch}
	resChan := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		s, err := newCudaState()
		resChan <- err
		if err != nil {
			return
		}
		defer s.destroy()
		cudaLoopMain(ch, s)
	}()
	if err := <-resChan; err != nil {
		return nil, err
	}
	return res, nil
}

// Close ends the loop and destroys its resources.
// This should only be called once.
func (c *cudaLoop) Close() {
	close(c.ch)
}

// Run evaluates the function on the CUDA thread and waits
// for the function to complete.
func (c *cudaLoop) Run(f func()) {
	res := make(chan struct{}, 1)
	c.ch <- &cudaLoopMsg{
		doneChan: res,
		f:        f,
	}
	<-res
}

// RunCUBLAS is like Run, but the function is given access
// to a cuBLAS handle.
func (c *cudaLoop) RunCUBLAS(f func(h C.cublasHandle_t)) {
	res := make(chan struct{}, 1)
	c.ch <- &cudaLoopMsg{
		doneChan: res,
		cublasF:  f,
	}
	<-res
}

type cudaLoopMsg struct {
	doneChan chan<- struct{}
	f        func()
	cublasF  func(h C.cublasHandle_t)
}

type cudaState struct {
	blas C.cublasHandle_t
	ctx  C.CUcontext
}

func newCudaState() (*cudaState, error) {
	C.cuInit(0)
	var dev C.CUdevice
	if C.cuDeviceGet((*C.CUdevice)(&dev), 0) != C.cuSuccess {
		return nil, ErrGetDevice
	}
	var ctx C.CUcontext
	if C.cuCtxCreate((*C.CUcontext)(&ctx), C.uint(0), dev) != C.cuSuccess {
		return nil, ErrMakeContext
	}

	handle := C.anyvec_cuda_new_handle()
	if C.anyvec_cuda_is_null(unsafe.Pointer(handle)) != C.int(0) {
		C.cuCtxDestroy(ctx)
		return nil, ErrMakeHandle
	}

	return &cudaState{
		blas: handle,
		ctx:  ctx,
	}, nil
}

func (c *cudaState) destroy() {
	C.cublasDestroy(c.blas)
	C.cuCtxDestroy(c.ctx)
}

func cudaLoopMain(ch <-chan *cudaLoopMsg, s *cudaState) {
	for msg := range ch {
		if msg.f != nil {
			msg.f()
		} else {
			msg.cublasF(s.blas)
		}
		close(msg.doneChan)
	}
}

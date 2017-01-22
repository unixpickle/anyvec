//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"

cublasOperation_t noTranspose = CUBLAS_OP_N;
cublasOperation_t transpose = CUBLAS_OP_T;
cublasSideMode_t sideModeRight = CUBLAS_SIDE_RIGHT;
cublasSideMode_t sideModeLeft = CUBLAS_SIDE_LEFT;
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

type compareType int

const (
	lessThan compareType = iota
	greaterThan
	equalTo
)

// A Handle manages an internal CUDA context.
type Handle struct {
	gc      *gc
	pool    *bufferPool
	loop    *cudaLoop
	kernels *mathKernels
	rand    *randomizer
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	err := createMainLoop()
	if err != nil {
		return nil, err
	}
	res := &Handle{gc: newGC(), pool: newBufferPool(), loop: getMainLoop()}
	runtime.SetFinalizer(res, func(obj *Handle) {
		obj.loop.Run(func() {
			if obj.kernels != nil {
				obj.kernels.Destroy()
			}
			if obj.rand != nil {
				obj.rand.Destroy()
			}
			obj.pool.Destroy()
		})
	})
	return res, nil
}

// SetPoolSize tells the Handle how much CUDA memory it
// should cache, in bytes.
//
// The buffer pool is useful for applications that tend to
// allocate vectors of the same length over and over.
// For applications that allocate many different vector
// sizes, the buffer pool will not help very much.
//
// The default pool size is 0.
func (h *Handle) SetPoolSize(size int64) {
	h.pool.SetCapacity(size)
}

func (h *Handle) runWithKernels(f func()) {
	h.runWithKernelsOpt(false, f)
}

func (h *Handle) runWithKernelsAsync(f func()) {
	h.runWithKernelsOpt(true, f)
}

func (h *Handle) runWithRandAsync(f func()) {
	h.loop.RunAsync(func() {
		var err error
		if h.kernels == nil {
			h.kernels, err = newMathKernels()
			if err != nil {
				panic(err)
			}
		}
		if h.rand == nil {
			h.rand, err = newRandomizer()
			if err != nil {
				panic(err)
			}
		}
		f()
	})
}

func (h *Handle) runWithKernelsOpt(async bool, f func()) {
	loopFunc := func() {
		var err error
		if h.kernels == nil {
			h.kernels, err = newMathKernels()
			if err != nil {
				panic(err)
			}
		}
		f()
	}
	if async {
		h.loop.RunAsync(loopFunc)
	} else {
		h.loop.Run(loopFunc)
	}
}

// A buffer is an on-device memory buffer.
type buffer struct {
	handle *Handle
	size   int
	ptr    unsafe.Pointer
}

// newBufferPtr creates a buffer around an existing piece
// of device memory.
// The inLoop argument indicates whether or not the call
// is being made from the handle's loop.
func newBufferPtr(h *Handle, size int, buf unsafe.Pointer, inLoop bool) *buffer {
	res := &buffer{
		handle: h,
		size:   size,
		ptr:    buf,
	}
	if inLoop {
		h.gc.Alloc(size)
	} else {
		h.loop.Run(func() {
			h.gc.Alloc(size)
		})
	}
	runtime.SetFinalizer(res, func(b *buffer) {
		b.handle.loop.Run(func() {
			h.pool.Free(b.ptr, b.size)
			h.gc.Free(b.size)
		})
	})
	return res
}

// newBuffer allocates a buffer.
func newBuffer(h *Handle, size int) (res *buffer, err error) {
	h.loop.Run(func() {
		var buf unsafe.Pointer
		buf, err = h.pool.Alloc(size)
		if err == nil {
			res = newBufferPtr(h, size, buf, true)
		}
	})
	if err != nil {
		return nil, err
	}
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
			err = cudaError("cudaMemcpy", C.cudaMemcpy(dest, x.ptr, C.size_t(x.size),
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
	var err error
	b.handle.loop.Run(func() {
		err = cudaError("cudaMemset", C.cudaMemset(b.ptr, 0, C.size_t(b.size)))
	})
	runtime.KeepAlive(b)
	return err
}

// Set copies the contents of a buffer into b.
func (b *buffer) Set(b1 *buffer) error {
	if b1.size != b.size {
		return errors.New("buffer sizes do not match")
	}
	var res error
	b.handle.loop.Run(func() {
		res = cudaError("cudaMemcpy", C.cudaMemcpy(b.ptr, b1.ptr, C.size_t(b.size),
			C.cudaMemcpyDeviceToDevice))
	})
	runtime.KeepAlive(b1)
	runtime.KeepAlive(b)
	return res
}

// Set32 copies 32-bit floats into the buffer.
func (b *buffer) Set32(src []float32) error {
	res := b.hostToDevice(len(src)*4, unsafe.Pointer(&src[0]))
	runtime.KeepAlive(src)
	return res
}

// Set64 copies 64-bit floats into the buffer.
func (b *buffer) Set64(src []float64) error {
	res := b.hostToDevice(len(src)*8, unsafe.Pointer(&src[0]))
	runtime.KeepAlive(src)
	return res
}

// Get32 copies 32-bit floats out of the buffer.
func (b *buffer) Get32(dst []float32) error {
	res := b.deviceToHost(len(dst)*4, unsafe.Pointer(&dst[0]))
	runtime.KeepAlive(dst)
	return res
}

// Get64 copies 64-bit floats out of the buffer.
func (b *buffer) Get64(dst []float64) error {
	res := b.deviceToHost(len(dst)*8, unsafe.Pointer(&dst[0]))
	runtime.KeepAlive(dst)
	return res
}

func (b *buffer) hostToDevice(size int, src unsafe.Pointer) error {
	if size > b.size {
		panic("buffer overflow")
	}
	var res error
	b.handle.loop.Run(func() {
		res = cudaError("cudaMemcpy", C.cudaMemcpy(b.ptr, src, C.size_t(size),
			C.cudaMemcpyHostToDevice))
	})
	runtime.KeepAlive(b)
	return res
}

func (b *buffer) deviceToHost(size int, dst unsafe.Pointer) error {
	if size > b.size {
		panic("buffer overflow")
	}
	var res error
	b.handle.loop.Run(func() {
		res = cudaError("cudaMemcpy", C.cudaMemcpy(dst, b.ptr, C.size_t(size),
			C.cudaMemcpyDeviceToHost))
	})
	runtime.KeepAlive(b)
	return res
}

func blasTransposeOp(trans bool) C.cublasOperation_t {
	if trans {
		return C.transpose
	}
	return C.noTranspose
}

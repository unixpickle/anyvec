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
	gc        *gc
	allocator allocator
	loop      *cudaLoop
	kernels   *mathKernels
	rand      *randomizer
}

// NewHandle attempts to get a new Handle.
//
// This will attempt to allocate a lot of the available
// CUDA memory.
// To avoid such behavior, use NewHandleNoMem.
func NewHandle() (*Handle, error) {
	err := createMainLoop()
	if err != nil {
		return nil, err
	}
	var a allocator
	getMainLoop().Run(func() {
		a, err = newBuddyAllocator()
	})
	if err != nil {
		return nil, err
	}
	return &Handle{
		gc:        newGC(),
		allocator: a,
		loop:      getMainLoop(),
	}, nil
}

// NewHandleNoMem creates a Handle, but it does not seize
// CUDA memory.
//
// The resulting Handle will allocate and free memory
// dynamically, which could hinder performance.
func NewHandleNoMem() (*Handle, error) {
	err := createMainLoop()
	if err != nil {
		return nil, err
	}
	return &Handle{
		gc:        newGC(),
		allocator: directAllocator{},
		loop:      getMainLoop(),
	}, nil
}

// Close releases resources that the handle was holding.
//
// You should only call this when you are done using the
// handle and any creators or vectors depending on it.
//
// You should never call Close more than once.
func (h *Handle) Close() {
	h.loop.Run(func() {
		if h.kernels != nil {
			h.kernels.Destroy()
			h.kernels = nil
		}
		if h.rand != nil {
			h.rand.Destroy()
			h.rand = nil
		}
		h.allocator.Destroy()
	})
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
			h.kernels, err = newMathKernels(h.allocator)
			if err != nil {
				panic(err)
			}
		}
		if h.rand == nil {
			h.rand, err = newRandomizer(h.allocator)
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
			h.kernels, err = newMathKernels(h.allocator)
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
		b.handle.loop.RunAsync(func() {
			h.allocator.Free(b.ptr)
			h.gc.Free(b.size)
		})
	})
	return res
}

// newBuffer allocates a buffer.
func newBuffer(h *Handle, size int) (res *buffer, err error) {
	h.loop.Run(func() {
		var buf unsafe.Pointer
		buf, err = h.allocator.Alloc(size)
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
func (b *buffer) Set(offInDest int, b1 *buffer) error {
	if offInDest <= -b1.size {
		return nil
	}

	b1Size := b1.size
	if offInDest > b.size {
		return nil
	} else if b1Size+offInDest > b.size {
		b1Size = b.size - offInDest
	}

	var res error
	b.handle.loop.Run(func() {
		var bOffPtr, b1OffPtr unsafe.Pointer
		totalSize := b1Size
		if offInDest < 0 {
			totalSize += offInDest
			b1OffPtr = unsafe.Pointer(uintptr(b1.ptr) + uintptr(-offInDest))
			bOffPtr = b.ptr
		} else {
			b1OffPtr = b1.ptr
			bOffPtr = unsafe.Pointer(uintptr(b.ptr) + uintptr(offInDest))
		}
		res = cudaError("cudaMemcpy", C.cudaMemcpy(bOffPtr, b1OffPtr, C.size_t(totalSize),
			C.cudaMemcpyDeviceToDevice))
	})
	runtime.KeepAlive(b1)
	runtime.KeepAlive(b)
	return res
}

// SetInts copies 32-bit integers into the buffer.
func (b *buffer) SetInts(table []int) error {
	cints := make([]C.int, len(table))
	for i, x := range table {
		cints[i] = C.int(x)
	}
	res := b.hostToDevice(len(table)*4, unsafe.Pointer(&cints[0]))
	runtime.KeepAlive(cints)
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

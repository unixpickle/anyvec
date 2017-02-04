//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
*/
import "C"

import (
	"errors"
	"unsafe"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/memalloc"
)

const (
	minAllocatorSize = 1 << 20
	maxAllocators    = 3
	allocAlignment   = 32
	allocHeadroom    = 1 << 25
)

type allocator interface {
	Alloc(size int) (unsafe.Pointer, error)
	Free(ptr unsafe.Pointer)
	Destroy()
}

type directAllocator struct{}

func (d directAllocator) Alloc(size int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	err := cudaError("cudaMalloc", C.cudaMalloc(&ptr, C.size_t(size)))
	return ptr, err
}

func (d directAllocator) Free(ptr unsafe.Pointer) {
	C.cudaFree(ptr)
}

func (d directAllocator) Destroy() {
}

// A bfcAllocator allocates memory using the best-fit with
// coalescing allocation algorithm.
type bfcAllocator struct {
	destroyed  bool
	allocators []*memalloc.MemAllocator
}

func newBuddyAllocator() (*bfcAllocator, error) {
	var free, total C.size_t
	err := cudaError("cudaGetMemInfo", C.cudaMemGetInfo(&free, &total))
	if err != nil {
		return nil, essentials.AddCtx("initialize allocator", err)
	}
	free -= allocHeadroom

	res := &bfcAllocator{}
	for len(res.allocators) < maxAllocators && free >= minAllocatorSize {
		amount := greatestTwoPower(free)
		var region unsafe.Pointer
		err = cudaError("cudaMalloc", C.cudaMalloc(&region, amount))
		if err != nil {
			free /= 2
			continue
		}
		res.allocators = append(res.allocators, &memalloc.MemAllocator{
			Start:     region,
			Size:      int(amount),
			Allocator: memalloc.NewBFC(int(amount), allocAlignment),
		})
		free -= amount
	}
	if len(res.allocators) == 0 {
		return nil, errors.New("initialize allocator: not enough memory")
	}
	return res, nil
}

func (b *bfcAllocator) Alloc(size int) (unsafe.Pointer, error) {
	if b.destroyed {
		panic("alloc: allocator was destroyed")
	}
	for _, x := range b.allocators {
		ptr, err := x.Alloc(size)
		if err == nil {
			return ptr, nil
		}
	}
	return nil, errors.New("alloc: out of memory")
}

func (b *bfcAllocator) Free(ptr unsafe.Pointer) {
	// Can happen if the handle is closed before a buffer is
	// garbage collected.
	if b.destroyed {
		return
	}

	for _, x := range b.allocators {
		if x.Contains(ptr) {
			x.Free(ptr)
			return
		}
	}

	panic("invalid pointer was freed")
}

func (b *bfcAllocator) Destroy() {
	if b.destroyed {
		panic("buddy allocator already destroyed")
	}
	b.destroyed = true
	for _, x := range b.allocators {
		C.cudaFree(unsafe.Pointer(x.Start))
	}
}

func greatestTwoPower(amount C.size_t) C.size_t {
	res := C.size_t(1)
	for res*2 < amount && res*2 > res {
		res *= 2
	}
	return res
}

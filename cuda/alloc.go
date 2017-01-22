//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
*/
import "C"

import "unsafe"

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

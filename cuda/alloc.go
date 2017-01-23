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
)

const (
	minBuddySize      = 1 << 20
	maxBuddyRoots     = 3
	minBuddyAllocSize = 32
	buddyHeadroom     = 1 << 25
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

// A buddyAllocator allocates memory using the buddy
// allocation algorithm.
type buddyAllocator struct {
	destroyed bool
	nodes     []*buddyNode
}

func newBuddyAllocator() (*buddyAllocator, error) {
	var free, total C.size_t
	err := cudaError("cudaGetMemInfo", C.cudaMemGetInfo(&free, &total))
	if err != nil {
		return nil, err
	}
	free -= buddyHeadroom

	res := &buddyAllocator{}
	for len(res.nodes) < maxBuddyRoots && free >= minBuddySize {
		amount := greatestTwoPower(free)
		var buddyMem unsafe.Pointer
		err = cudaError("cudaMalloc", C.cudaMalloc(&buddyMem, amount))
		if err != nil {
			free /= 2
			continue
		}
		node := newBuddyNode(buddyMem, unsafe.Pointer(uintptr(buddyMem)+uintptr(amount)))
		res.nodes = append(res.nodes, node)
		free -= amount
	}
	if len(res.nodes) == 0 {
		return nil, errors.New("failed to allocate buddy nodes")
	}
	return res, nil
}

func greatestTwoPower(amount C.size_t) C.size_t {
	res := C.size_t(1)
	for res*2 < amount {
		res *= 2
	}
	return res
}

func (b *buddyAllocator) Alloc(size int) (unsafe.Pointer, error) {
	if b.destroyed {
		panic("alloc from destroyed allocator")
	}
	if size < minBuddyAllocSize {
		size = minBuddyAllocSize
	}
	for _, x := range b.nodes {
		ptr, err := x.Alloc(uintptr(size))
		if err == nil {
			return unsafe.Pointer(ptr), nil
		}
	}
	return nil, errors.New("no free memory nodes")
}

func (b *buddyAllocator) Free(ptr unsafe.Pointer) {
	// Can happen if the handle is closed before a buffer is
	// garbage collected.
	if b.destroyed {
		return
	}

	for _, x := range b.nodes {
		if uintptr(x.start) <= uintptr(ptr) && uintptr(x.end) > uintptr(ptr) {
			x.Free(ptr)
			return
		}
	}

	panic("invalid pointer was freed")
}

func (b *buddyAllocator) Destroy() {
	if b.destroyed {
		panic("buddy allocator already destroyed")
	}
	b.destroyed = true
	for _, x := range b.nodes {
		C.cudaFree(unsafe.Pointer(x.start))
	}
}

// buddyNode is a node in a buddy allocator tree.
type buddyNode struct {
	start unsafe.Pointer
	end   unsafe.Pointer

	// biggestFree is the biggest free chunk of memory within
	// this node.
	biggestFree uintptr

	// children, or nil if the node is full or empty.
	left  *buddyNode
	right *buddyNode
}

func newBuddyNode(start, end unsafe.Pointer) *buddyNode {
	return &buddyNode{
		start:       start,
		end:         end,
		biggestFree: uintptr(end) - uintptr(start),
	}
}

func (b *buddyNode) Alloc(size uintptr) (unsafe.Pointer, error) {
	if size == 0 {
		panic("zero allocs not allowed")
	}
	if size > b.biggestFree {
		return nil, errors.New("allocator node is full")
	}

	if size > b.size()/2 {
		if b.left != nil || b.right != nil {
			panic("cannot have children")
		}
		b.biggestFree = 0
		return b.start, nil
	}

	if b.left == nil {
		b.split()
	}

	subPtr, err := b.left.Alloc(size)
	if err == nil {
		b.updateBiggestFree()
		return subPtr, nil
	}
	subPtr, err = b.right.Alloc(size)
	if err == nil {
		b.updateBiggestFree()
		return subPtr, nil
	}

	panic("impossible situation")
}

func (b *buddyNode) Free(p unsafe.Pointer) {
	if b.left == nil && b.right == nil {
		if p != b.start {
			panic("freed invalid address")
		}
		b.biggestFree = b.size()
		return
	}
	if uintptr(p) < uintptr(b.right.start) {
		b.left.Free(p)
	} else {
		b.right.Free(p)
	}
	if b.left.biggestFree+b.right.biggestFree == b.size() {
		b.biggestFree = b.size()
		b.left = nil
		b.right = nil
	} else {
		b.updateBiggestFree()
	}
}

func (b *buddyNode) split() {
	if b.left != nil || b.right != nil {
		panic("cannot split node")
	}
	halfSize := b.size() / 2
	b.left = &buddyNode{
		start:       b.start,
		end:         unsafe.Pointer(uintptr(b.start) + halfSize),
		biggestFree: halfSize,
	}
	b.right = &buddyNode{
		start:       unsafe.Pointer(uintptr(b.start) + halfSize),
		end:         b.end,
		biggestFree: halfSize,
	}
}

func (b *buddyNode) updateBiggestFree() {
	if b.left == nil || b.right == nil {
		panic("missing children")
	}
	if b.left.biggestFree < b.right.biggestFree {
		b.biggestFree = b.right.biggestFree
	} else {
		b.biggestFree = b.left.biggestFree
	}
}

func (b *buddyNode) size() uintptr {
	return uintptr(b.end) - uintptr(b.start)
}

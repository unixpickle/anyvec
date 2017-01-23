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

func (b *buddyAllocator) Free(ptr unsafe.Pointer) {
	// Can happen if the handle is closed before a buffer is
	// garbage collected.
	if b.destroyed {
		return
	}
}

func (b *buddyAllocator) Destroy() {
	if b.destroyed {
		panic("buddy allocator already destroyed")
	}
	b.destroyed = true
	for _, x := range b.nodes {
	}
}

// buddyNode is a node in a buddy allocator tree.
type buddyNode struct {
	start uintptr
	end   uintptr

	// biggestFree is the biggest free chunk of memory within
	// this node.
	biggestFree uintptr

	// children, or nil if the node is full or empty.
	left  *buddyNode
	right *buddyNode
}

func newBuddyNode(start, end uintptr) *buddyNode {
	return &buddyNode{
		start:       start,
		end:         end,
		biggestFree: end - start,
	}
}

func (b *buddyNode) Alloc(size uintptr) (uintptr, error) {
	if size == 0 {
		panic("zero allocs not allowed")
	}
	if size > b.biggestFree {
		return 0, errors.New("allocator node is full")
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

func (b *buddyNode) Free(p uintptr) {
	if b.left == nil && b.right == nil {
		if p != b.start {
			panic("freed invalid address")
		}
		b.biggestFree = b.size()
		return
	}
	if p < b.right.start {
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
		end:         b.start + halfSize,
		biggestFree: halfSize,
	}
	b.right = &buddyNode{
		start:       b.start + halfSize,
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
	return b.end - b.start
}

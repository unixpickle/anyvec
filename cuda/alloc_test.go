//+build !nocuda

package cuda

import (
	"testing"
	"unsafe"
)

func TestBuddyAllocations(t *testing.T) {
	tempSlice := make([]byte, 65)
	b := newBuddyNode(unsafe.Pointer(&tempSlice[0]), unsafe.Pointer(&tempSlice[64]))
	base := b.start

	ptr := buddyAllocMust(b.Alloc(4))
	if uintptr(ptr) != uintptr(base) {
		t.Fatalf("expected 0 but got %v", uintptr(ptr)-uintptr(base))
	}
	_, err := b.Alloc(64)
	if err == nil {
		t.Fatal("should have failed")
	}
	ptr2 := buddyAllocMust(b.Alloc(8))
	if uintptr(ptr2) != uintptr(base)+8 {
		t.Fatalf("expected 8 but got %v", uintptr(ptr2)-uintptr(base))
	}

	ptr3 := buddyAllocMust(b.Alloc(4))
	if uintptr(ptr3) != uintptr(base)+4 {
		t.Fatalf("expected 4 but got %v", uintptr(ptr3)-uintptr(base))
	}
	b.Free(ptr)
	b.Free(ptr2)
	ptr = buddyAllocMust(b.Alloc(1))
	if uintptr(ptr) != uintptr(base) {
		t.Fatalf("expected 0 but got %v", uintptr(ptr)-uintptr(base))
	}
	b.Free(ptr)
	ptr = buddyAllocMust(b.Alloc(8))
	if uintptr(ptr) != uintptr(base)+8 {
		t.Fatalf("expected 8 but got %v", uintptr(ptr)-uintptr(base))
	}
	b.Free(ptr3)
	ptr3 = buddyAllocMust(b.Alloc(8))
	if uintptr(ptr3) != uintptr(base) {
		t.Fatalf("expected 0 but got %v", uintptr(ptr3)-uintptr(base))
	}
	b.Free(ptr)
	b.Free(ptr3)

	ptr = buddyAllocMust(b.Alloc(64))
	if uintptr(ptr) != uintptr(base) {
		t.Fatalf("expected 0 but got %v", uintptr(ptr)-uintptr(base))
	}

	_, err = b.Alloc(1)
	if err == nil {
		t.Fatal("should have failed")
	}
}

func buddyAllocMust(ptr unsafe.Pointer, err error) unsafe.Pointer {
	if err != nil {
		panic(err)
	}
	return ptr
}

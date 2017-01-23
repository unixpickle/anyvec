//+build !nocuda

package cuda

import "testing"

func TestBuddyAllocations(t *testing.T) {
	b := newBuddyNode(8, 8+64)
	ptr := buddyAllocMust(b.Alloc(4))
	if ptr != 8 {
		t.Fatalf("expected 8 but got %v", ptr)
	}
	_, err := b.Alloc(64)
	if err == nil {
		t.Fatal("should have failed")
	}
	ptr2 := buddyAllocMust(b.Alloc(8))
	if ptr2 != 16 {
		t.Fatalf("expected 16 but got %v", ptr2)
	}

	ptr3 := buddyAllocMust(b.Alloc(4))
	if ptr3 != 12 {
		t.Fatalf("expected 12 but got %v", ptr3)
	}
	b.Free(ptr)
	b.Free(ptr2)
	ptr = buddyAllocMust(b.Alloc(1))
	if ptr != 8 {
		t.Fatalf("expected 8 but got %v", ptr)
	}
	b.Free(ptr)
	ptr = buddyAllocMust(b.Alloc(8))
	if ptr != 16 {
		t.Fatalf("expected 16 but got %v", ptr)
	}
	b.Free(ptr3)
	ptr3 = buddyAllocMust(b.Alloc(8))
	if ptr3 != 8 {
		t.Fatalf("expected 8 but got %v", ptr3)
	}
	b.Free(ptr)
	b.Free(ptr3)

	ptr = buddyAllocMust(b.Alloc(64))
	if ptr != 8 {
		t.Fatalf("expected 8 but got %v", ptr)
	}

	_, err = b.Alloc(1)
	if err == nil {
		t.Fatal("should have failed")
	}
}

func buddyAllocMust(ptr uintptr, err error) uintptr {
	if err != nil {
		panic(err)
	}
	return ptr
}

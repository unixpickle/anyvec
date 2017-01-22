package cuda

import (
	"testing"

	"github.com/unixpickle/anyvec/anyvectest"
)

func TestVector32(t *testing.T) {
	h, err := NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	tester := anyvectest.Tester{
		Creator: NewCreator32(h),
	}
	tester.TestAll(t)
}

func TestVector32Pool(t *testing.T) {
	h, err := NewHandle()
	if err != nil {
		t.Fatal(err)
	}
	h.SetPoolSize(65536)
	tester := anyvectest.Tester{
		Creator: NewCreator32(h),
	}
	tester.TestAll(t)
}

func BenchmarkVector32(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	bencher := anyvectest.Bencher{
		Creator: NewCreator32(h),
	}
	bencher.BenchmarkAll(b)
}

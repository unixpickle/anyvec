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
	tester := anyvectest.Tester32{
		Creator: NewCreator32(h),
	}
	tester.TestAll(t)
}

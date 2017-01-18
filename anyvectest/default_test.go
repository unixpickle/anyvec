package anyvectest

import (
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
)

func TestDefault32(t *testing.T) {
	tester := Tester{Creator: anyvec32.DefaultCreator{}}
	tester.TestAll(t)
}

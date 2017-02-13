package anyvectest

import (
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestDefault32(t *testing.T) {
	tester := Tester{Creator: anyvec32.DefaultCreator{}}
	tester.TestAll(t)
}

func TestDefault64(t *testing.T) {
	tester := Tester{Creator: anyvec64.DefaultCreator{}}
	tester.TestAll(t)
}

func BenchmarkDefault32(b *testing.B) {
	bencher := Bencher{Creator: anyvec32.DefaultCreator{}}
	bencher.BenchmarkAll(b)
}

func BenchmarkDefault64(b *testing.B) {
	bencher := Bencher{Creator: anyvec32.DefaultCreator{}}
	bencher.BenchmarkAll(b)
}

package cuda

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
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

func BenchmarkGemmOneVec(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	mat := &anyvec32.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	vec := &anyvec32.Matrix{
		Data: randomVector(c, 300),
		Rows: 300,
		Cols: 1,
	}
	prod := &anyvec32.Matrix{
		Data: c.MakeVector(300),
		Rows: 300,
		Cols: 1,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prod.Product(false, false, 1, mat, vec, 0)
	}
}

func BenchmarkGemmMat(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	mat1 := &anyvec32.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	mat2 := &anyvec32.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	prod := &anyvec32.Matrix{
		Data: c.MakeVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prod.Product(false, false, 1, mat1, mat2, 0)
	}
}

func randomVector(c anyvec32.Creator, size int) anyvec32.Vector {
	d := make([]float32, size)
	for i := range d {
		d[i] = float32(rand.NormFloat64())
	}
	return c.MakeVectorData(d)
}

package cuda

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/anyvec"
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

func BenchmarkGemmOneVec(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	mat := &anyvec.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	vec := &anyvec.Matrix{
		Data: randomVector(c, 300),
		Rows: 300,
		Cols: 1,
	}
	prod := &anyvec.Matrix{
		Data: c.MakeVector(300),
		Rows: 300,
		Cols: 1,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prod.Product(false, false, float32(1), mat, vec, float32(0))
	}
}

func BenchmarkGemmMat(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	mat1 := &anyvec.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	mat2 := &anyvec.Matrix{
		Data: randomVector(c, 300*300),
		Rows: 300,
		Cols: 300,
	}
	prod := &anyvec.Matrix{
		Data: c.MakeVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prod.Product(false, false, float32(1), mat1, mat2, float32(0))
	}
}

func BenchmarkMul(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	v1 := randomVector(c, 1024)
	scale := make([]float32, v1.Len())
	for i := range scale {
		scale[i] = rand.Float32()*0.005 + 0.9975
	}
	v2 := c.MakeVectorData(scale)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v1.Mul(v2)
	}
}

func BenchmarkDiv(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	v1 := randomVector(c, 1024)
	scale := make([]float32, v1.Len())
	for i := range scale {
		scale[i] = rand.Float32()*0.005 + 0.9975
	}
	v2 := c.MakeVectorData(scale)

	// Initialize kernels.
	v1.Div(v2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v1.Div(v2)
	}
}

func BenchmarkAddScaler(b *testing.B) {
	benchmarkIter(1024, b, func(v anyvec.Vector) {
		v.AddScaler(float32(2.0))
	})
}

func BenchmarkAddRepeated(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	v := randomVector(c, 256*128)
	v1 := randomVector(c, 256)

	// Initialize kernels.
	anyvec.AddRepeated(v, v1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anyvec.AddRepeated(v, v1)
	}
}

func BenchmarkExp(b *testing.B) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	vec := make([]float32, 1024)
	for i := range vec {
		// Nice choice of values since -e^vec[i] = vec[i]
		vec[i] = -0.56714329041
	}
	v := c.MakeVectorData(vec)

	// Initialize kernels.
	anyvec.Exp(v)
	v.Scale(float32(-1))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Prevent numbers from exploding or changing too much.
		anyvec.Exp(v)
		v.Scale(float32(-1))
	}
}

func BenchmarkTanh(b *testing.B) {
	benchmarkIter(1024, b, anyvec.Tanh)
}

func BenchmarkClipPos(b *testing.B) {
	benchmarkIter(1024, b, anyvec.ClipPos)
}

func BenchmarkNormDist(b *testing.B) {
	benchmarkIter(1024, b, func(v anyvec.Vector) {
		anyvec.Rand(v, anyvec.Normal, nil)
	})
}

func BenchmarkLogSoftmax(b *testing.B) {
	benchmarkIter(256*64, b, func(v anyvec.Vector) {
		anyvec.LogSoftmax(v, 256)
	})
}

func BenchmarkAddLogs(b *testing.B) {
	benchmarkIter(256*64, b, func(v anyvec.Vector) {
		anyvec.AddLogs(v, 256)
	})
}

func benchmarkIter(size int, b *testing.B, f func(anyvec.Vector)) {
	h, err := NewHandle()
	if err != nil {
		b.Fatal(err)
	}
	c := NewCreator32(h)
	vec := make([]float32, 1024)
	for i := range vec {
		vec[i] = float32(rand.NormFloat64())
	}
	v := c.MakeVectorData(vec)

	// Initialize kernels.
	f(v)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f(v)
	}
}

func randomVector(c anyvec.Creator, size int) anyvec.Vector {
	d := make([]float32, size)
	for i := range d {
		d[i] = float32(rand.NormFloat64())
	}
	return c.MakeVectorData(d)
}

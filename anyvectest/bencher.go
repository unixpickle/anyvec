package anyvectest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/anyvec"
)

// A Bencher measures the performance of various anyvec
// functions.
type Bencher struct {
	Creator anyvec.Creator
}

// BenchmarkAll runs all of the benchmarks.
func (b *Bencher) BenchmarkAll(bench *testing.B) {
	bench.Run("GemmOneVec", b.BenchmarkGemmOneVec)
	bench.Run("GemmMat", b.BenchmarkGemmMat)
	bench.Run("Mul", b.BenchmarkMul)
	bench.Run("Div", b.BenchmarkDiv)
	bench.Run("AddScaler", b.BenchmarkAddScaler)
	bench.Run("AddRepeated", b.BenchmarkAddRepeated)
	bench.Run("Exp", b.BenchmarkExp)
	bench.Run("Tanh", b.BenchmarkTanh)
	bench.Run("ClipPos", b.BenchmarkClipPos)
	bench.Run("NormDist", b.BenchmarkNormDist)
	bench.Run("LogSoftmax", b.BenchmarkLogSoftmax)
	bench.Run("AddLogs", b.BenchmarkAddLogs)
}

func (b *Bencher) BenchmarkGemmOneVec(bench *testing.B) {
	mat := &anyvec.Matrix{
		Data: b.randomVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	vec := &anyvec.Matrix{
		Data: b.randomVector(300),
		Rows: 300,
		Cols: 1,
	}
	prod := &anyvec.Matrix{
		Data: b.Creator.MakeVector(300),
		Rows: 300,
		Cols: 1,
	}
	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		prod.Product(false, false, float32(1), mat, vec, float32(0))
	}
	// Force lazy operations to finish.
	prod.Data.Data()
}

func (b *Bencher) BenchmarkGemmMat(bench *testing.B) {
	mat1 := &anyvec.Matrix{
		Data: b.randomVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	mat2 := &anyvec.Matrix{
		Data: b.randomVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	prod := &anyvec.Matrix{
		Data: b.Creator.MakeVector(300 * 300),
		Rows: 300,
		Cols: 300,
	}
	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		prod.Product(false, false, float32(1), mat1, mat2, float32(0))
	}
	// Force lazy operations to finish.
	prod.Data.Data()
}

func (b *Bencher) BenchmarkMul(bench *testing.B) {
	v1 := b.randomVector(1024)
	scale := make([]float64, v1.Len())
	for i := range scale {
		scale[i] = rand.Float64()*0.005 + 0.9975
	}
	v2 := b.Creator.MakeVectorData(b.Creator.MakeNumericList(scale))
	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		v1.Mul(v2)
	}
	// Force lazy operations to finish.
	v1.Data()
}

func (b *Bencher) BenchmarkDiv(bench *testing.B) {
	v1 := b.randomVector(1024)
	scale := make([]float64, v1.Len())
	for i := range scale {
		scale[i] = rand.Float64()*0.005 + 0.9975
	}
	v2 := b.Creator.MakeVectorData(b.Creator.MakeNumericList(scale))

	// Initialize kernels.
	v1.Div(v2)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		v1.Div(v2)
	}
	// Force lazy operations to finish.
	v1.Data()
}

func (b *Bencher) BenchmarkAddScaler(bench *testing.B) {
	b.benchmarkIter(1024, bench, func(v anyvec.Vector) {
		v.AddScaler(float32(2.0))
	})
}

func (b *Bencher) BenchmarkAddRepeated(bench *testing.B) {
	v := b.randomVector(256 * 128)
	v1 := b.randomVector(256)

	// Initialize kernels.
	anyvec.AddRepeated(v, v1)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		anyvec.AddRepeated(v, v1)
	}
	// Force lazy operations to finish.
	v.Data()
}

func (b *Bencher) BenchmarkExp(bench *testing.B) {
	vec := make([]float64, 1024)
	for i := range vec {
		// Nice choice of values since -e^vec[i] = vec[i]
		vec[i] = -0.56714329041
	}
	v := b.Creator.MakeVectorData(b.Creator.MakeNumericList(vec))

	// Initialize kernels.
	anyvec.Exp(v)
	v.Scale(float32(-1))

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		// Prevent numbers from exploding or changing too much.
		anyvec.Exp(v)
		v.Scale(float32(-1))
	}
	// Force lazy operations to finish.
	v.Data()
}

func (b *Bencher) BenchmarkTanh(bench *testing.B) {
	b.benchmarkIter(1024, bench, anyvec.Tanh)
}

func (b *Bencher) BenchmarkClipPos(bench *testing.B) {
	b.benchmarkIter(1024, bench, anyvec.ClipPos)
}

func (b *Bencher) BenchmarkNormDist(bench *testing.B) {
	b.benchmarkIter(1024, bench, func(v anyvec.Vector) {
		anyvec.Rand(v, anyvec.Normal, nil)
	})
}

func (b *Bencher) BenchmarkLogSoftmax(bench *testing.B) {
	b.benchmarkIter(256*64, bench, func(v anyvec.Vector) {
		anyvec.LogSoftmax(v, 256)
	})
}

func (b *Bencher) BenchmarkAddLogs(bench *testing.B) {
	b.benchmarkIter(256*64, bench, func(v anyvec.Vector) {
		anyvec.AddLogs(v, 256)
	})
}

func (b *Bencher) benchmarkIter(size int, bench *testing.B, f func(anyvec.Vector)) {
	v := b.randomVector(size)

	// Initialize kernels.
	f(v)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		f(v)
	}
	// Force lazy operations to finish.
	v.Data()
}

func (b *Bencher) randomVector(size int) anyvec.Vector {
	d := make([]float64, size)
	for i := range d {
		d[i] = rand.NormFloat64()
	}
	return b.Creator.MakeVectorData(b.Creator.MakeNumericList(d))
}

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
	bench.Run("GemmBatch", b.BenchmarkGemmBatch)
	bench.Run("Mul", b.BenchmarkMul)
	bench.Run("Div", b.BenchmarkDiv)
	bench.Run("AddScalar", b.BenchmarkAddScalar)
	bench.Run("AddRepeated", b.BenchmarkAddRepeated)
	bench.Run("SumRows", b.BenchmarkSumRows)
	bench.Run("Exp", b.BenchmarkExp)
	bench.Run("Tanh", b.BenchmarkTanh)
	bench.Run("ClipPos", b.BenchmarkClipPos)
	bench.Run("NormDist", b.BenchmarkNormDist)
	bench.Run("LogSoftmax", b.BenchmarkLogSoftmax)
	bench.Run("AddLogs", b.BenchmarkAddLogs)
	bench.Run("Map", b.BenchmarkMap)
	bench.Run("MapTranspose", b.BenchmarkMapTranspose)
	bench.Run("MapMax", b.BenchmarkMapMax)
	bench.Run("Pow", b.BenchmarkPow)
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

func (b *Bencher) BenchmarkGemmBatch(bench *testing.B) {
	const batchSize = 8
	mat1 := &anyvec.MatrixBatch{
		Data: b.randomVector(128 * 256 * batchSize),
		Rows: 256,
		Cols: 128,
		Num:  batchSize,
	}
	mat2 := &anyvec.MatrixBatch{
		Data: b.randomVector(128 * 4 * batchSize),
		Rows: 128,
		Cols: 4,
		Num:  batchSize,
	}
	prod := &anyvec.MatrixBatch{
		Data: b.Creator.MakeVector(256 * 4 * batchSize),
		Rows: 256,
		Cols: 4,
		Num:  batchSize,
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

func (b *Bencher) BenchmarkAddScalar(bench *testing.B) {
	b.benchmarkIter(1024, bench, func(v anyvec.Vector) {
		v.AddScalar(float32(2.0))
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

func (b *Bencher) BenchmarkSumRows(bench *testing.B) {
	v := b.randomVector(256 * 128)

	// Initialize kernels.
	anyvec.SumRows(v, 256)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		anyvec.SumRows(v, 256)
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

func (b *Bencher) BenchmarkMap(bench *testing.B) {
	in, out, m := b.mapperTestData()

	// Initialize kernels.
	m.Map(in, out)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		m.Map(in, out)
	}
	// Trigger lazy evaluations.
	out.Data()
}

func (b *Bencher) BenchmarkMapTranspose(bench *testing.B) {
	in, out, m := b.mapperTestData()

	// Initialize kernels.
	m.MapTranspose(out, in)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		m.MapTranspose(out, in)
	}
	// Trigger lazy evaluations.
	out.Data()
}

func (b *Bencher) BenchmarkMapMax(bench *testing.B) {
	b.benchmarkIter(4096, bench, func(v anyvec.Vector) {
		anyvec.MapMax(v, 4)
	})
}

func (b *Bencher) BenchmarkPow(bench *testing.B) {
	names := []string{"Square", "Sqrt", "InvSquare", "InvSqrt", "Unit"}
	pows := []float64{2, 0.5, -2, -0.5, 1}
	for i, name := range names {
		bench.Run(name, func(bench *testing.B) {
			b.benchmarkIter(4096, bench, func(v anyvec.Vector) {
				anyvec.Pow(v, v.Creator().MakeNumeric(pows[i]))
			})
		})
	}
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

func (b *Bencher) mapperTestData() (in, out anyvec.Vector, m anyvec.Mapper) {
	in = b.randomVector(4096)
	out = b.randomVector(4096 * 2)
	mapping := make([]int, out.Len())
	for i := range mapping {
		mapping[i] = rand.Intn(in.Len())
	}
	m = b.Creator.MakeMapper(in.Len(), mapping)
	return
}

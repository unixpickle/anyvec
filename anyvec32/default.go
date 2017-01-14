package anyvec32

import (
	"math"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
)

// DefaultCreator is the built-in Creator.
// It provides a basic on-CPU implementation.
type DefaultCreator struct{}

// MakeVector creates a zero vector.
func (d DefaultCreator) MakeVector(size int) Vector {
	return make(vector, size)
}

// MakeVectorData creates a copy of the data.
func (d DefaultCreator) MakeVectorData(data []float32) Vector {
	return append(vector{}, data...)
}

type vector []float32

func (v vector) Len() int {
	return len(v)
}

func (v vector) Data() []float32 {
	return append([]float32{}, v...)
}

func (v vector) SetData(v1 []float32) {
	copy(v, v1)
}

func (v vector) Copy() Vector {
	return append(vector{}, v...)
}

func (v vector) Scale(s float32) {
	blas32.Scal(len(v), s, v.blasVec())
}

func (v vector) AddScaler(s float32) {
	for i := range v {
		v[i] += s
	}
}

func (v vector) Dot(v1 Vector) float32 {
	return blas32.Dot(len(v), v.blasVec(), v1.(vector).blasVec())
}

func (v vector) Add(v1 Vector) {
	blas32.Axpy(len(v), 1, v1.(vector).blasVec(), v.blasVec())
}

func (v vector) Sub(v1 Vector) {
	blas32.Axpy(len(v), -1, v1.(vector).blasVec(), v.blasVec())
}

func (v vector) Mul(v1 Vector) {
	for i, x := range v1.(vector) {
		v[i] *= x
	}
}

func (v vector) Div(v1 Vector) {
	for i, x := range v1.(vector) {
		v[i] /= x
	}
}

func (v vector) Gemm(transA, transB bool, m, n, k int, alpha float32, a Vector, lda int, b Vector,
	ldb int, beta float32, ldc int) {
	tA := blas.NoTrans
	tB := blas.NoTrans
	if transA {
		tA = blas.Trans
	}
	if transB {
		tB = blas.Trans
	}
	blas32.Implementation().Sgemm(tA, tB, m, n, k, alpha, a.(vector), lda, b.(vector),
		ldb, beta, v, ldc)
}

func (v vector) Exp() {
	for i, x := range v {
		v[i] = float32(math.Exp(float64(x)))
	}
}

func (v vector) blasVec() blas32.Vector {
	return blas32.Vector{Data: v, Inc: 1}
}

package anyvec32

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
	"github.com/unixpickle/anyvec"
)

// DefaultCreator is the built-in Creator.
// It provides a basic on-CPU implementation.
type DefaultCreator struct{}

// MakeNumeric creates a float32.
func (d DefaultCreator) MakeNumeric(x float64) anyvec.Numeric {
	return float32(x)
}

// MakeNumericList creates a []float32.
func (d DefaultCreator) MakeNumericList(x []float64) anyvec.NumericList {
	res := make([]float32, len(x))
	for i, k := range x {
		res[i] = float32(k)
	}
	return res
}

// MakeVector creates a zero vector.
func (d DefaultCreator) MakeVector(size int) anyvec.Vector {
	res := make(vector, size)
	return &res
}

// MakeVectorData creates a copy of the data.
func (d DefaultCreator) MakeVectorData(data anyvec.NumericList) anyvec.Vector {
	res := append(vector{}, data.([]float32)...)
	return &res
}

// Concat concatenates vectors.
func (d DefaultCreator) Concat(vs ...anyvec.Vector) anyvec.Vector {
	var res vector
	for _, v := range vs {
		res = append(res, *v.(*vector)...)
	}
	return &res
}

// MakeMapper creates a new Mapper.
func (d DefaultCreator) MakeMapper(inSize int, table []int) anyvec.Mapper {
	return &mapper{
		table:  table,
		inSize: inSize,
	}
}

type vector []float32

func (v *vector) Creator() anyvec.Creator {
	return DefaultCreator{}
}

func (v *vector) Len() int {
	return len(*v)
}

func (v *vector) Data() anyvec.NumericList {
	return append([]float32{}, *v...)
}

func (v *vector) SetData(v1 anyvec.NumericList) {
	copy(*v, v1.([]float32))
}

func (v *vector) Set(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	copy(*v, *v1.(*vector))
}

func (v *vector) Copy() anyvec.Vector {
	res := append(vector{}, *v...)
	return &res
}

func (v *vector) Slice(start, end int) anyvec.Vector {
	res := append(vector{}, (*v)[start:end]...)
	return &res
}

func (v *vector) SetSlice(start int, v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	if start < 0 {
		if -start < v1.Len() {
			copy(*v, (*v1.(*vector))[-start:])
		}
	} else {
		copy((*v)[start:], *v1.(*vector))
	}
}

func (v *vector) Scale(s anyvec.Numeric) {
	blas32.Scal(len(*v), s.(float32), v.blasVec())
}

func (v *vector) AddScaler(s anyvec.Numeric) {
	sNum := s.(float32)
	for i := range *v {
		(*v)[i] += sNum
	}
}

func (v *vector) Dot(v1 anyvec.Vector) anyvec.Numeric {
	return blas32.Dot(len(*v), v.blasVec(), v1.(*vector).blasVec())
}

func (v *vector) Add(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	blas32.Axpy(len(*v), 1, v1.(*vector).blasVec(), v.blasVec())
}

func (v *vector) Sub(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	blas32.Axpy(len(*v), -1, v1.(*vector).blasVec(), v.blasVec())
}

func (v *vector) Mul(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	for i, x := range *v1.(*vector) {
		(*v)[i] *= x
	}
}

func (v *vector) Div(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	for i, x := range *v1.(*vector) {
		(*v)[i] /= x
	}
}

func (v *vector) Gemm(transA, transB bool, m, n, k int, alpha anyvec.Numeric, a anyvec.Vector,
	lda int, b anyvec.Vector, ldb int, beta anyvec.Numeric, ldc int) {
	if v == a || v == b {
		panic("output cannot equal inputs")
	}
	tA := blas.NoTrans
	tB := blas.NoTrans
	if transA {
		tA = blas.Trans
	}
	if transB {
		tB = blas.Trans
	}
	blas32.Implementation().Sgemm(tA, tB, m, n, k, alpha.(float32), *a.(*vector), lda,
		*b.(*vector), ldb, beta.(float32), *v, ldc)
}

func (v *vector) blasVec() blas32.Vector {
	return blas32.Vector{Data: *v, Inc: 1}
}

type mapper struct {
	table  []int
	inSize int
}

func (r *mapper) Creator() anyvec.Creator {
	return DefaultCreator{}
}

func (r *mapper) InSize() int {
	return r.inSize
}

func (r *mapper) OutSize() int {
	return len(r.table)
}

func (r *mapper) Map(in, out anyvec.Vector) {
	if in == out {
		panic("arguments cannot be equal")
	} else if in.Len() != r.inSize {
		panic("bad input dimensions")
	} else if out.Len() != len(r.table) {
		panic("bad output dimensions")
	}
	inV := in.(*vector)
	outV := out.(*vector)
	for i, x := range r.table {
		(*outV)[i] = (*inV)[x]
	}
}

func (r *mapper) MapTranspose(in, out anyvec.Vector) {
	if in == out {
		panic("arguments cannot be equal")
	} else if in.Len() != len(r.table) {
		panic("bad input dimensions")
	} else if out.Len() != r.inSize {
		panic("bad output dimensions")
	}
	inV := in.(*vector)
	outV := out.(*vector)
	for i, x := range r.table {
		(*outV)[x] += (*inV)[i]
	}
}

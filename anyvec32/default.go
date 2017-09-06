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
	res := make([]float32, size)
	return &vector{backing: &res, start: 0, slice: res}
}

// MakeVectorData creates a copy of the data.
func (d DefaultCreator) MakeVectorData(data anyvec.NumericList) anyvec.Vector {
	res := append([]float32{}, data.([]float32)...)
	return &vector{backing: &res, start: 0, slice: res}
}

// Concat concatenates vectors.
func (d DefaultCreator) Concat(vs ...anyvec.Vector) anyvec.Vector {
	cap := 0
	for _, v := range vs {
		cap += v.Len()
	}
	res := make([]float32, 0, cap)
	for _, v := range vs {
		res = append(res, v.(*vector).slice...)
	}
	return &vector{backing: &res, start: 0, slice: res}
}

// MakeMapper creates a new Mapper.
func (d DefaultCreator) MakeMapper(inSize int, table []int) anyvec.Mapper {
	return &mapper{
		table:  table,
		inSize: inSize,
	}
}

// NumOps returns a new NumOps object.
func (d DefaultCreator) NumOps() anyvec.NumOps {
	return NumOps{}
}

// Float64 converts a float32 to a float64.
func (d DefaultCreator) Float64(n anyvec.Numeric) float64 {
	return float64(n.(float32))
}

// Float64Slice converts a []float32 to a []float64.
func (d DefaultCreator) Float64Slice(n anyvec.NumericList) []float64 {
	list := n.([]float32)
	res := make([]float64, len(list))
	for i, f := range list {
		res[i] = float64(f)
	}
	return res
}

type vector struct {
	// Information used to check for overlap.
	backing *[]float32
	start   int

	slice []float32
}

func (v *vector) Creator() anyvec.Creator {
	return DefaultCreator{}
}

func (v *vector) Len() int {
	return len(v.slice)
}

func (v *vector) Overlaps(v1 anyvec.Vector) bool {
	v1Vec := v1.(*vector)
	return v1Vec.backing == v.backing &&
		v.start < v1Vec.start+len(v1Vec.slice) &&
		v1Vec.start < v.start+len(v.slice)
}

func (v *vector) Data() anyvec.NumericList {
	return append([]float32{}, v.slice...)
}

func (v *vector) SetData(v1 anyvec.NumericList) {
	copy(v.slice, v1.([]float32))
}

func (v *vector) Set(v1 anyvec.Vector) {
	if v.Overlaps(v1) {
		panic("invalid overlap")
	}
	copy(v.slice, v1.(*vector).slice)
}

func (v *vector) Copy() anyvec.Vector {
	return v.Creator().Concat(v)
}

func (v *vector) Slice(start, end int) anyvec.Vector {
	return &vector{
		backing: v.backing,
		start:   v.start + start,
		slice:   v.slice[start:end],
	}
}

func (v *vector) Scale(s anyvec.Numeric) {
	blas32.Scal(len(v.slice), s.(float32), v.blasVec())
}

func (v *vector) AddScalar(s anyvec.Numeric) {
	sNum := s.(float32)
	for i := range v.slice {
		v.slice[i] += sNum
	}
}

func (v *vector) Dot(v1 anyvec.Vector) anyvec.Numeric {
	return blas32.Dot(v.Len(), v.blasVec(), v1.(*vector).blasVec())
}

func (v *vector) Add(v1 anyvec.Vector) {
	if v.Overlaps(v1) {
		panic("invalid overlap")
	}
	blas32.Axpy(v.Len(), 1, v1.(*vector).blasVec(), v.blasVec())
}

func (v *vector) Sub(v1 anyvec.Vector) {
	if v.Overlaps(v1) {
		panic("invalid overlap")
	}
	blas32.Axpy(v.Len(), -1, v1.(*vector).blasVec(), v.blasVec())
}

func (v *vector) Mul(v1 anyvec.Vector) {
	if v.Overlaps(v1) {
		panic("invalid overlap")
	}
	for i, x := range v1.(*vector).slice {
		v.slice[i] *= x
	}
}

func (v *vector) Div(v1 anyvec.Vector) {
	if v.Overlaps(v1) {
		panic("invalid overlap")
	}
	for i, x := range v1.(*vector).slice {
		v.slice[i] /= x
	}
}

func (v *vector) Gemm(transA, transB bool, m, n, k int, alpha anyvec.Numeric, a anyvec.Vector,
	lda int, b anyvec.Vector, ldb int, beta anyvec.Numeric, ldc int) {
	if v.Overlaps(a) || v.Overlaps(b) {
		panic("invalid overlap")
	}
	tA := blas.NoTrans
	tB := blas.NoTrans
	if transA {
		tA = blas.Trans
	}
	if transB {
		tB = blas.Trans
	}
	blas32.Implementation().Sgemm(tA, tB, m, n, k, alpha.(float32), a.(*vector).slice,
		lda, b.(*vector).slice, ldb, beta.(float32), v.slice, ldc)
}

func (v *vector) blasVec() blas32.Vector {
	return blas32.Vector{Data: v.slice, Inc: 1}
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
	if in.Overlaps(out) {
		panic("invalid overlap")
	} else if in.Len() != r.inSize {
		panic("bad input dimensions")
	} else if out.Len() != len(r.table) {
		panic("bad output dimensions")
	}
	inV := in.(*vector)
	outV := out.(*vector)
	for i, x := range r.table {
		outV.slice[i] = inV.slice[x]
	}
}

func (r *mapper) MapTranspose(in, out anyvec.Vector) {
	if in.Overlaps(out) {
		panic("invalid overlap")
	} else if in.Len() != len(r.table) {
		panic("bad input dimensions")
	} else if out.Len() != r.inSize {
		panic("bad output dimensions")
	}
	inV := in.(*vector)
	outV := out.(*vector)
	for i, x := range r.table {
		outV.slice[x] += inV.slice[i]
	}
}

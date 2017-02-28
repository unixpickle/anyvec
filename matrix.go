package anyvec

import (
	"fmt"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
	"github.com/gonum/blas/blas64"
)

// Matrix is a helper for performing matrix operations.
type Matrix struct {
	Data Vector
	Rows int
	Cols int
}

// Product sets m = alpha*a*b + beta*m.
// If transA is set, then a is transposed.
// If transB is set, then b is transposed.
func (m *Matrix) Product(transA, transB bool, alpha Numeric, a, b *Matrix, beta Numeric) {
	x, n, k := a.Rows, m.Cols, a.Cols
	if transA {
		x, k = k, x
	}
	if transB {
		n = b.Rows
	}
	m.Data.Gemm(transA, transB, x, n, k, alpha, a.Data, a.Cols, b.Data, b.Cols, beta,
		m.Cols)
}

// Transpose stores the transpose of src in m.
func (m *Matrix) Transpose(src *Matrix) {
	if src.Rows != m.Cols || src.Cols != m.Rows {
		panic("invalid output dimensions")
	}
	Transpose(src.Data, m.Data, src.Rows)
}

// A Transposer is a vector which can treat itself as a
// matrix and compute its transpose.
// In order for the Transposer to know how to lay out its
// values, the number of rows in the matrix must be
// specified.
//
// A transpose is out-of place.
// The receiver should not equal the output vector.
type Transposer interface {
	Transpose(out Vector, inRows int)
}

// Transpose treats v as a matrix and transposes it,
// saving the result to out.
//
// The inRows argument specifies the number of rows in the
// input (row-major) matrix.
// It must divide v.Len().
//
// If v does not implement Transposer, a default
// implementation is used.
//
// v and out should not be equal.
func Transpose(v, out Vector, inRows int) {
	if t, ok := v.(Transposer); ok {
		t.Transpose(out, inRows)
	} else {
		if v.Len()%inRows != 0 {
			panic("row count must divide vector length")
		}
		cols := v.Len() / inRows
		mapping := make([]int, 0, inRows*cols)
		for destRow := 0; destRow < cols; destRow++ {
			for destCol := 0; destCol < inRows; destCol++ {
				sourceIdx := destRow + destCol*cols
				mapping = append(mapping, sourceIdx)
			}
		}
		m := v.Creator().MakeMapper(inRows*cols, mapping)
		m.Map(v, out)
	}
}

// A Gemver is a vector which can set itself to a
// matrix-vector product.
//
// Specifically, a Gemver implements the BLAS gemv API
// with itself as the destination vector.
type Gemver interface {
	Gemv(trans bool, m, n int, alpha Numeric, matA Vector, lda int,
		x Vector, incx int, beta Numeric, incy int)
}

// Gemv computes a matrix-vector product and adds it to a
// scaler multiple of y.
//
// If y does not implement Gemver, a default
// implementation is used which supports float32 and
// float64 numeric types.
func Gemv(trans bool, m, n int, alpha Numeric, matA Vector, lda int,
	x Vector, incx int, beta Numeric, y Vector, incy int) {
	if g, ok := y.(Gemver); ok {
		g.Gemv(trans, m, n, alpha, matA, lda, x, incx, beta, incy)
		return
	}

	tA := blas.NoTrans
	if trans {
		tA = blas.Trans
	}

	switch yData := y.Data().(type) {
	case []float32:
		blas32.Implementation().Sgemv(tA, m, n,
			alpha.(float32),
			matA.Data().([]float32), lda,
			x.Data().([]float32), incx,
			beta.(float32),
			yData, incy)
		y.SetData(yData)
	case []float64:
		blas64.Implementation().Dgemv(tA, m, n,
			alpha.(float64),
			matA.Data().([]float64), lda,
			x.Data().([]float64), incx,
			beta.(float64),
			yData, incy)
		y.SetData(yData)
	default:
		panic(fmt.Sprintf("unsupported type: %T", yData))
	}
}

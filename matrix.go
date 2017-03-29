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
	opARows, opACols := a.Rows, a.Cols
	opBRows, opBCols := b.Rows, b.Cols
	if transA {
		opARows, opACols = opACols, opARows
	}
	if transB {
		opBRows, opBCols = opBCols, opBRows
	}
	if opARows != m.Rows || opBCols != m.Cols || opACols != opBRows {
		panic("matrix dimension mismatch")
	}

	x, n, k := a.Rows, m.Cols, a.Cols
	if transA {
		x, k = k, x
	}
	if transB {
		n = b.Rows
	}
	Gemm(transA, transB, x, n, k, alpha, a.Data, a.Cols, b.Data, b.Cols, beta, m.Data, m.Cols)
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
//
// In general, the receiver (y) cannot be equal to either
// of the two operands (x or a).
type Gemver interface {
	Gemv(trans bool, m, n int, alpha Numeric, a Vector, lda int,
		x Vector, incx int, beta Numeric, incy int)
}

// Gemv computes a matrix-vector product.
//
// If y does not implement Gemver, a default
// implementation is used which supports float32 and
// float64 numeric types.
func Gemv(trans bool, m, n int, alpha Numeric, a Vector, lda int,
	x Vector, incx int, beta Numeric, y Vector, incy int) {
	if g, ok := y.(Gemver); ok {
		g.Gemv(trans, m, n, alpha, a, lda, x, incx, beta, incy)
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
			a.Data().([]float32), lda,
			x.Data().([]float32), incx,
			beta.(float32),
			yData, incy)
		y.SetData(yData)
	case []float64:
		blas64.Implementation().Dgemv(tA, m, n,
			alpha.(float64),
			a.Data().([]float64), lda,
			x.Data().([]float64), incx,
			beta.(float64),
			yData, incy)
		y.SetData(yData)
	default:
		panic(fmt.Sprintf("unsupported type: %T", yData))
	}
}

// A Gemmer is a vector capable of setting itself to a
// matrix-matrix product.
//
// Specifically, a Gemmer implements the BLAS gemm API.
//
// In general, the receiver (c) cannot be equal to either
// of the two operands (a or b).
type Gemmer interface {
	Gemm(transA, transB bool, m, n, k int, alpha Numeric, a Vector, lda int,
		b Vector, ldb int, beta Numeric, ldc int)
}

// Gemm computes a matrix-matrix product.
//
// If c does not implement Gemmer, a default
// implementation is used which supports float32 and
// float64 numeric types.
func Gemm(transA, transB bool, m, n, k int, alpha Numeric, a Vector, lda int,
	b Vector, ldb int, beta Numeric, c Vector, ldc int) {
	if g, ok := c.(Gemmer); ok {
		g.Gemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, ldc)
		return
	}

	tA, tB := blas.NoTrans, blas.NoTrans
	if transA {
		tA = blas.Trans
	}
	if transB {
		tB = blas.Trans
	}

	switch cData := c.Data().(type) {
	case []float32:
		blas32.Implementation().Sgemm(tA, tB, m, n, k,
			alpha.(float32),
			a.Data().([]float32), lda,
			b.Data().([]float32), ldb,
			beta.(float32),
			cData, ldc)
		c.SetData(cData)
	case []float64:
		blas64.Implementation().Dgemm(tA, tB, m, n, k,
			alpha.(float64),
			a.Data().([]float64), lda,
			b.Data().([]float64), ldb,
			beta.(float64),
			cData, ldc)
		c.SetData(cData)
	default:
		panic(fmt.Sprintf("unsupported type: %T", cData))
	}
}

// MatrixBatch is a helper for batches of matrices.
//
// Each matrix is row-major, and the matrices are packed
// one after another.
type MatrixBatch struct {
	Data Vector
	Num  int
	Rows int
	Cols int
}

// Product performs batch matrix multiplication.
// This is a batched version of Matrix.Product().
func (m *MatrixBatch) Product(transA, transB bool, alpha Numeric, a, b *MatrixBatch,
	beta Numeric) {
	if m.Num != a.Num || a.Num != b.Num {
		panic("batch size mismatch")
	}

	if m.Num == 1 {
		mat1 := &Matrix{Data: m.Data, Rows: m.Rows, Cols: m.Cols}
		mat2 := &Matrix{Data: a.Data, Rows: a.Rows, Cols: a.Cols}
		mat3 := &Matrix{Data: b.Data, Rows: b.Rows, Cols: b.Cols}
		mat1.Product(transA, transB, alpha, mat2, mat3, beta)
		return
	}

	opARows, opACols := a.Rows, a.Cols
	opBRows, opBCols := b.Rows, b.Cols
	if transA {
		opARows, opACols = opACols, opARows
	}
	if transB {
		opBRows, opBCols = opBCols, opBRows
	}
	if opARows != m.Rows || opBCols != m.Cols || opACols != opBRows {
		panic("matrix dimension mismatch")
	}

	x, n, k := a.Rows, m.Cols, a.Cols
	if transA {
		x, k = k, x
	}
	if transB {
		n = b.Rows
	}
	BatchedGemm(transA, transB, m.Num, x, n, k, alpha, a.Data, b.Data, beta, m.Data)
}

// BatchedGemmer is a vector capable of performing batched
// matrix multiplications.
//
// This is similar to a Gemmer, but with batching and
// without support for custom leading dimensions.
type BatchedGemmer interface {
	BatchedGemm(transA, transB bool, num, m, n, k int, alpha Numeric,
		a, b Vector, beta Numeric)
}

// BatchedGemm performs batched matrix multiplication.
//
// If c does not implement BatchedGemmer, a default
// implementation is used which relies on Gemm.
func BatchedGemm(transA, transB bool, num, m, n, k int, alpha Numeric,
	a, b Vector, beta Numeric, c Vector) {
	if bg, ok := c.(BatchedGemmer); ok {
		bg.BatchedGemm(transA, transB, num, m, n, k, alpha, a, b, beta)
		return
	}

	aBatch := splitBatch(a, num)
	bBatch := splitBatch(b, num)
	cBatch := splitBatch(c, num)

	var offset int
	for i, subC := range cBatch {
		lda, ldb := k, n
		if transA {
			lda = m
		}
		if transB {
			ldb = k
		}
		Gemm(transA, transB, m, n, k, alpha, aBatch[i], lda, bBatch[i], ldb, beta, subC, n)
		c.SetSlice(offset, subC)
		offset += subC.Len()
	}
}

func splitBatch(vec Vector, n int) []Vector {
	if vec.Len()%n != 0 {
		panic("vector length not divisible by batch size")
	}
	var res []Vector
	chunkSize := vec.Len() / n
	for i := 0; i < n; i++ {
		chunk := vec.Slice(chunkSize*i, chunkSize*(i+1))
		res = append(res, chunk)
	}
	return res
}

package anyvec32

// Matrix is a helper for performing matrix operations.
type Matrix struct {
	Data Vector
	Rows int
	Cols int
}

// Product sets m = alpha*a*b + beta*m.
// If transA is set, then a is transposed.
// If transB is set, then b is transposed.
func (m *Matrix) Product(transA, transB bool, alpha float32, a, b *Matrix, beta float32) {
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

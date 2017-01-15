package anyvec32

// A Vector is the main primitive for manipulating 32-bit
// floating-point values.
//
// Vectors are only thread-safe for reading.
// Any time a vector is being modified, it should not be
// accessed from another thread.
//
// A Vector can only be used with other Vector instances
// created by the same Creator.
type Vector interface {
	// Len returns the size of the vector.
	Len() int

	// Data returns a copy of the vector's contents.
	Data() []float32

	// SetData copies the values from v into the receiver.
	//
	// It must be the case that len(v) <= receiver.Len().
	SetData(v []float32)

	// Copy creates a copy of the vector.
	Copy() Vector

	// Scale scales the vector by a constant.
	Scale(s float32)

	// AddScaler adds a scaler to every component.
	AddScaler(s float32)

	// Dot computes the dot product with another vector.
	Dot(v Vector) float32

	// Add adds another vector to this vector.
	Add(v Vector)

	// Sub subtracts a vector from this vector.
	Sub(v Vector)

	// Mul multiplies the components of the vector by the
	// components of v and stores the result in the receiver.
	Mul(v Vector)

	// Div divides the components of the vector by the
	// components of v and stores the result in the receiver.
	Div(v Vector)

	// Gemm performs a matrix multiplication and stores the
	// result in the receiver.
	//
	// The method signature matches the BLAS standard, where
	// the output matrix c is the receiver.
	//
	// Row-major order is used.
	Gemm(transA, transB bool, m, n, k int, alpha float32, a Vector, lda int, b Vector,
		ldb int, beta float32, ldc int)
}

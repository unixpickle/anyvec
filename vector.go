package anyvec

// A Numeric is some numerical type, such as a float32 or
// a float64.
type Numeric interface{}

// A NumericList is a slice of numerical values, such as a
// []float32 or a []float64.
type NumericList interface{}

// A Vector is the main primitive for storing values.
//
// Vectors are only thread-safe for reading.
// Any time a vector is being modified, it should not be
// accessed from another thread.
//
// A Vector can only be used with other Vector instances
// created by the same Creator.
type Vector interface {
	// Creator returns the creator of the vector.
	Creator() Creator

	// Len returns the number of vector components.
	Len() int

	// Data returns a copy of the vector's contents.
	Data() NumericList

	// SetData copies the values from v into the receiver.
	//
	// It must be the case that len(v) <= receiver.Len().
	SetData(v NumericList)

	// Copy creates a copy of the vector.
	Copy() Vector

	// Slice creates a copy of a subset of the vector.
	//
	// The start argument specifies the first element.
	// The end argument specifies the element after the last
	// element, so that (end-start) is the length of the
	// resulting vector.
	// The indices must be within the bounds of the vector.
	//
	// The result should contain its own copy of the data,
	// unlike how slices work in Go.
	Slice(start, end int) Vector

	// Scale scales the vector by a constant.
	Scale(s Numeric)

	// AddScaler adds a scaler to every component.
	AddScaler(s Numeric)

	// Dot computes the dot product with another vector.
	Dot(v Vector) Numeric

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
	Gemm(transA, transB bool, m, n, k int, alpha Numeric, a Vector, lda int, b Vector,
		ldb int, beta Numeric, ldc int)
}

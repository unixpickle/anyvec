package anyvec

// A Numeric is some numerical type, such as a float32 or
// a float64.
type Numeric interface{}

// A NumericList is a list of numerical values, such as a
// []float32 or a []float64.
// It needn't be a native slice.
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

	// Overlaps checks if the receiver has an overlapping
	// backing buffer with v.
	Overlaps(v Vector) bool

	// Data returns a copy of the vector's contents.
	Data() NumericList

	// SetData copies the values from v into the receiver.
	//
	// It must be the case that len(v) <= receiver.Len().
	SetData(v NumericList)

	// Set copies the contents of v into the receiver.
	// The receiver must not overlap with v.
	Set(v Vector)

	// Copy creates a copy of the vector.
	Copy() Vector

	// Slice creates a reference to a sub-range of the
	// vector.
	//
	// The start argument specifies the first element.
	// The end argument specifies the element after the last
	// element, so that (end-start) is the length of the
	// resulting vector.
	// The indices must be within the bounds of the vector.
	//
	// The result is backed by the same buffer.
	Slice(start, end int) Vector

	// Scale scales the vector by a constant.
	Scale(s Numeric)

	// AddScalar adds a scaler to every component.
	AddScalar(s Numeric)

	// Dot computes the dot product with another vector.
	Dot(v Vector) Numeric

	// Add adds another vector to this vector.
	// The receiver must not overlap with v.
	Add(v Vector)

	// Sub subtracts a vector from this vector.
	// The receiver must not overlap with v.
	Sub(v Vector)

	// Mul multiplies the components of the vector by the
	// components of v and stores the result in the receiver.
	// The receiver must not overlap with v.
	Mul(v Vector)

	// Div divides the components of the vector by the
	// components of v and stores the result in the receiver.
	// The receiver must not overlap with v.
	Div(v Vector)
}

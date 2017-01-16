package anyvec

// A Creator creates Vectors and Numerics.
// Creators should be comparable using the == operation,
// meaning that they should typically be pointers.
type Creator interface {
	// MakeNumeric creates a Numeric from a 64-bit floating
	// point constant.
	MakeNumeric(x float64) Numeric

	// MakeNumericList creates a NumericList from a slice of
	// 64-bit floating point constants.
	MakeNumericList(x []float64) NumericList

	// MakeVector creates a Vector full of zero values.
	MakeVector(size int) Vector

	// MakeVectorData creates a Vector containing the
	// contents of the list.
	MakeVectorData(data NumericList) Vector

	// Concat concatenates Vectors.
	// The resulting vector should contain a complete copy of
	// the original vectors, so that the original vectors can
	// be modified without changing the new vector.
	Concat(vs ...Vector) Vector
}

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

	// MakeMapper creates a Mapper based on the lookup table.
	//
	// Values in the table correspond to input indices.
	// Indices in the table correspond to output indices.
	//
	// No value in the table can be less than 0 or greater
	// than inSize-1.
	MakeMapper(inSize int, table []int) Mapper

	// NumOps returns a NumOps for the Numeric type.
	NumOps() NumOps

	// Float64 converts a Numeric back to a native float.
	// This may be a lossy conversion.
	Float64(n Numeric) float64

	// Float64Slice converts a NumericList back to a slice
	// of native floats.
	// This may be a lossy conversion.
	Float64Slice(n NumericList) []float64
}

// Ones creates a vector of all ones.
func Ones(c Creator, size int) Vector {
	res := c.MakeVector(size)
	res.AddScalar(c.MakeNumeric(1))
	return res
}

// Make creates a vector directly from a []float64.
// It combines the MakeVectorData and MakeNumericList
// methods of c.
func Make(c Creator, f []float64) Vector {
	return c.MakeVectorData(c.MakeNumericList(f))
}

// OneHot creates a one-hot vector.
//
// The vector has dim elements, and the i-th one is set.
func OneHot(c Creator, dim, i int) Vector {
	res := c.MakeVector(dim)
	res.Slice(i, i+1).SetData(c.MakeNumericList([]float64{1}))
	return res
}

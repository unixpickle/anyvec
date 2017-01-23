package anyvec

// A Mapper generates output vectors from input vectors
// by permuting, repeating, and possibly dropping input
// components.
// In particular, each output component takes its value
// from a pre-determined index in the input.
//
// A Mapper can be thought of as a matrix in which each
// row contains a single 1.
// When this matrix is multiplied by a column vector, it
// is like applying Map.
// When the transpose of the matrix is multiplied by a
// column vector, that is like applying ReverseMap.
//
// A Mapper has an exact input and output vector size.
// Mappers do not work with dynamic vector lengths.
type Mapper interface {
	// Creator returns the creator of the instance.
	Creator() Creator

	// InSize returns the size of inputs to Map.
	InSize() int

	// OutSize returns the size of outputs from Map.
	OutSize() int

	// Map applies the mapping to produce a new vector.
	Map(in, out Vector)

	// MapTranspose applies the transpose of the mapping.
	//
	// To each component of out[i], MapTranspose adds in[j]
	// for all j such that in[i] would map to out[j] in Map.
	//
	// Note that the result is added to the output vector,
	// rather than the output vector being overwritten.
	// To circumvent this, you can manually scale the output
	// vector by a factor of 0 before calling MapTranspose.
	MapTranspose(in, out Vector)
}

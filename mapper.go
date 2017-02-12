package anyvec

import "fmt"

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
	//
	// The input and output vectors cannot be equal.
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
	//
	// The input and output vectors cannot be equal.
	MapTranspose(in, out Vector)
}

// A MaxMapper can generate a Mapper that maps the maximum
// element of each of its rows to one element of a target.
// In other words, each element of the mapper's output
// corresponds to a different row in the MaxMapper, and
// specifically maps from the maximum element in that row.
type MaxMapper interface {
	MapMax(cols int) Mapper
}

// MapMax treats v like a matrix and creates a Mapper that
// maps each row to the maximum element of that row.
// If the vector does not implement MaxMapper, a default
// implementation is used which supports float32 and
// float64.
//
// The cols argument must divide the length of v.
func MapMax(v Vector, cols int) Mapper {
	if m, ok := v.(MaxMapper); ok {
		return m.MapMax(cols)
	} else {
		var mapping []int
		switch data := v.Data().(type) {
		case []float32:
			mapping = mapMax32(data, cols)
		case []float64:
			mapping = mapMax64(data, cols)
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
		return v.Creator().MakeMapper(v.Len(), mapping)
	}
}

func mapMax32(data []float32, cols int) []int {
	if len(data)%cols != 0 {
		panic("column count must divide vector size")
	}
	res := make([]int, len(data)/cols)
	for rowIdx := range res {
		row := data[cols*rowIdx : cols*(rowIdx+1)]
		res[rowIdx] = maxIndex32(row) + cols*rowIdx
	}
	return res
}

func mapMax64(data []float64, cols int) []int {
	if len(data)%cols != 0 {
		panic("column count must divide vector size")
	}
	res := make([]int, len(data)/cols)
	for rowIdx := range res {
		row := data[cols*rowIdx : cols*(rowIdx+1)]
		res[rowIdx] = maxIndex64(row) + cols*rowIdx
	}
	return res
}

func maxIndex32(data []float32) int {
	if len(data) == 0 {
		return 0
	}
	maxVal := data[0]
	maxIdx := 0
	for i, x := range data {
		if x > maxVal {
			maxIdx = i
			maxVal = x
		}
	}
	return maxIdx
}

func maxIndex64(data []float64) int {
	if len(data) == 0 {
		return 0
	}
	maxVal := data[0]
	maxIdx := 0
	for i, x := range data {
		if x > maxVal {
			maxIdx = i
			maxVal = x
		}
	}
	return maxIdx
}

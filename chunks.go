package anyvec

import "fmt"

// A ChunkAdder can add a different scaler to each
// contiguous chunk of itself.
// It is equivalent to the pseudo-code:
//
//     chunkSize := v.Len() / scalers.Len()
//     v[i] += scalers[i / chunkSize];
//
// The length of the receiver must be divisible by the
// length of the scaler vector, since each chunk must be
// the same length.
type ChunkAdder interface {
	AddChunks(scalers Vector)
}

// AddChunks adds scalers to contiguous chunks of a
// vector.
// If the vector does not implement ChunkAdder, a default
// implementation is used which supports float32 and
// float64 values.
// v and scalers must not be equal.
func AddChunks(v, scalers Vector) {
	if c, ok := v.(ChunkAdder); ok {
		c.AddChunks(scalers)
	} else {
		if v.Len()%scalers.Len() != 0 {
			panic("scaler count must divide vector length")
		}
		data := v.Data()
		switch data := data.(type) {
		case []float32:
			applyAddChunks32(data, scalers.Data().([]float32))
		case []float64:
			applyAddChunks64(data, scalers.Data().([]float64))
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
		v.SetData(data)
	}
}

func applyAddChunks32(data, scalers []float32) {
	chunkSize := len(data) / len(scalers)
	for i := range data {
		data[i] += scalers[i/chunkSize]
	}
}

func applyAddChunks64(data, scalers []float64) {
	chunkSize := len(data) / len(scalers)
	for i := range data {
		data[i] += scalers[i/chunkSize]
	}
}

// A ChunkScaler can scale contiguous chunks of itself,
// each by differenent scalers.
// It is equivalent to the pseudo-code:
//
//     chunkSize := v.Len() / scalers.Len()
//     v[i] *= scalers[i / chunkSize];
//
// The length of the receiver must be divisible by the
// length of the scaler vector, since each chunk must be
// the same length.
type ChunkScaler interface {
	ScaleChunks(scalers Vector)
}

// ScaleChunks scales contiguous chunks of a vector.
// If the vector does not implement ChunkScaler, a default
// implementation is used which supports float32 and
// float64 values.
// v and scalers must not be equal.
func ScaleChunks(v, scalers Vector) {
	if c, ok := v.(ChunkScaler); ok {
		c.ScaleChunks(scalers)
	} else {
		if v.Len()%scalers.Len() != 0 {
			panic("scaler count must divide vector length")
		}
		data := v.Data()
		switch data := data.(type) {
		case []float32:
			applyScaleChunks32(data, scalers.Data().([]float32))
		case []float64:
			applyScaleChunks64(data, scalers.Data().([]float64))
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
		v.SetData(data)
	}
}

func applyScaleChunks32(data, scalers []float32) {
	chunkSize := len(data) / len(scalers)
	for i := range data {
		data[i] *= scalers[i/chunkSize]
	}
}

func applyScaleChunks64(data, scalers []float64) {
	chunkSize := len(data) / len(scalers)
	for i := range data {
		data[i] *= scalers[i/chunkSize]
	}
}

// A RepeatedAdder can add to itself the repeated contents
// of another vector.
// It is equivalent to the following pseudo-code:
//
//     v[i] += scalers[i%scalers.Len()]
//
// The length of the scalers will never be zero.
type RepeatedAdder interface {
	AddRepeated(scalers Vector)
}

// AddRepeated adds the repeated form of scalers to v.
// If the vector does not implement RepeatedAdder, a
// default implementation is used.
// v and scalers must not be equal.
func AddRepeated(v, scalers Vector) {
	if r, ok := v.(RepeatedAdder); ok {
		r.AddRepeated(scalers)
	} else {
		if scalers.Len() == 0 {
			panic("repeated vector cannot be empty")
		}
		var joinMe []Vector
		var joinLen int
		for joinLen < v.Len() {
			joinLen += scalers.Len()
			joinMe = append(joinMe, scalers)
		}
		v.Add(v.Creator().Concat(joinMe...).Slice(0, v.Len()))
	}
}

// A RepeatedScaler can scale to itself the repeated
// contents of another vector.
// It is equivalent to the following pseudo-code:
//
//     v[i] *= scalers[i%scalers.Len()]
//
// The length of the scalers will never be zero.
type RepeatedScaler interface {
	ScaleRepeated(scalers Vector)
}

// ScaleRepeated scales the repeated form of scalers to v.
// If the vector does not implement RepeatedScaler, a
// default implementation is used.
// v and scalers must not be equal.
func ScaleRepeated(v, scalers Vector) {
	if r, ok := v.(RepeatedScaler); ok {
		r.ScaleRepeated(scalers)
	} else {
		if scalers.Len() == 0 {
			panic("repeated vector cannot be empty")
		}
		var joinMe []Vector
		var joinLen int
		for joinLen < v.Len() {
			joinLen += scalers.Len()
			joinMe = append(joinMe, scalers)
		}
		v.Mul(v.Creator().Concat(joinMe...).Slice(0, v.Len()))
	}
}

package anyvec

import "fmt"

// A ChunkScaler can scale contiguous chunks of itself,
// each by differenent scalers.
// The length of the receiver must be divisible by the
// length of the scaler vector, since the chunk size is
// determined by dividing the receiver's length by the
// number of scalers.
// Each scaled chunk is the same length.
type ChunkScaler interface {
	ScaleChunks(scalers Vector)
}

// ScaleChunks scales contigous chunks of a vector.
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
type RepeatedAdder interface {
	AddRepeated(v Vector)
}

// AddRepeated adds the repeated form of v1 to v.
// It is equivalent to v[i] += v1[i%v1.Len()].
// If the vector does not implement RepeatedAdder, a
// default implementation is used.
// v and v1 must not be equal.
func AddRepeated(v, v1 Vector) {
	if r, ok := v.(RepeatedAdder); ok {
		r.AddRepeated(v1)
	} else {
		if v1.Len() == 0 {
			panic("repeated vector cannot be empty")
		}
		var joinMe []Vector
		var joinLen int
		for joinLen < v.Len() {
			joinLen += v1.Len()
			joinMe = append(joinMe, v1)
		}
		v.Add(v.Creator().Concat(joinMe...).Slice(0, v.Len()))
	}
}

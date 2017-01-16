package anyvec

import (
	"fmt"
	"math"
)

// A PosClipper can clip its values using max(0, x).
type PosClipper interface {
	ClipPos()
}

// ClipPos clips the vector entries to positive values.
// If the vector does not implement PosClipper, a default
// implementation is used which supports float32 and
// float64 values.
func ClipPos(v Vector) {
	if p, ok := v.(PosClipper); ok {
		p.ClipPos()
	} else {
		applyUnitary(v, func(arg float64) float64 {
			return math.Max(0, arg)
		})
	}
}

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

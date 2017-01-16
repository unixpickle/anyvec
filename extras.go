package anyvec

import (
	"fmt"
	"math"
)

// A Tanher can compute its own hyperbolic tangent.
type Tanher interface {
	Tanh()
}

// Tanh computes the in-place hyperbolic tangent of the
// vector.
// If the vector does not implement Tanher, a default
// implementation is used which supports float32 and
// float64 values.
func Tanh(v Vector) {
	if t, ok := v.(Tanher); ok {
		t.Tanh()
	} else {
		applyUnitary(v, math.Tanh)
	}
}

// A Siner can compute its sine from radians.
type Siner interface {
	Sin()
}

// Sin computes the in-place sin of the vector.
// The arguments to sin are in radians.
// If the vector does not implement Siner, a default
// implementation is used which supports float32 and
// float64 values.
func Sin(v Vector) {
	if s, ok := v.(Siner); ok {
		s.Sin()
	} else {
		applyUnitary(v, math.Sin)
	}
}

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

// An Exper can exponentiate its entries in base e.
type Exper interface {
	Exp()
}

// Exp exponentiates the vector entries in base e.
// If the vector does not implement Exper, a default
// implementation is used which supports float32 and
// float64 values.
func Exp(v Vector) {
	if e, ok := v.(Exper); ok {
		e.Exp()
	} else {
		applyUnitary(v, math.Exp)
	}
}

// A Summer can sum up its entries.
type Summer interface {
	Sum() Numeric
}

// Sum sums the vector entries.
// If the vector does not implement Summer, a default
// implementation is used which supports float32 and
// float64 values.
func Sum(v Vector) Numeric {
	if s, ok := v.(Summer); ok {
		return s.Sum()
	} else {
		return applyAggregate(v, func(data []float64) float64 {
			var sum float64
			for _, x := range data {
				sum += x
			}
			return sum
		})
	}
}

// A Maxer can compute its maximum entry.
// The return value is unspecified if the vector is empty.
type Maxer interface {
	Max() Numeric
}

// Max computes the maximum vector entry.
// If the vector does not implement Maxer, a default
// implementation is used which supports float32 and
// float64 values.
// The return value is unspecified if the vector is empty.
func Max(v Vector) Numeric {
	if m, ok := v.(Maxer); ok {
		return m.Max()
	} else {
		return applyAggregate(v, func(data []float64) float64 {
			if len(data) == 0 {
				return 0
			}
			max := data[0]
			for _, x := range data[1:] {
				if x > max {
					max = x
				}
			}
			return max
		})
	}
}

// A LogSoftmaxer computes the logarithm of the softmax of
// its components.
// Softmaxing is done in chunks, meaning that every
// chunkSize components are treated as their own vector
// and softmaxed in place.
// If the chunkSize is 0, then the entire vector is
// softmaxed as a single unit.
// The chunkSize should divide the vector evenly, unless
// it is zero.
type LogSoftmaxer interface {
	LogSoftmax(chunkSize int)
}

// LogSoftmax computes the logarithm of the softmax in
// place.
// If the vector does not implement LogSoftmaxer, a
// default implementation is used which supports float32
// and float64 values.
func LogSoftmax(v Vector, chunkSize int) {
	if l, ok := v.(LogSoftmaxer); ok {
		l.LogSoftmax(chunkSize)
	} else {
		if v.Len() == 0 {
			return
		} else if chunkSize == 0 {
			chunkSize = v.Len()
		} else if v.Len()%chunkSize != 0 {
			panic("chunkSize must divide vector length")
		}
		data := v.Data()
		switch data := data.(type) {
		case []float32:
			applyLogSoftmax32(data, chunkSize)
		case []float64:
			applyLogSoftmax64(data, chunkSize)
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
		v.SetData(data)
	}
}

func applyLogSoftmax32(data []float32, chunkSize int) {
	for i := 0; i < len(data); i += chunkSize {
		vec := data[i : i+chunkSize]

		max := float32(math.Inf(-1))
		for _, x := range vec {
			if x > max {
				max = x
			}
		}

		var expSum float64
		for _, x := range vec {
			expSum += math.Exp(float64(x - max))
		}
		sumLog := float32(math.Log(expSum)) + max

		for i, x := range vec {
			vec[i] = float32(x) - sumLog
		}
	}
}

func applyLogSoftmax64(data []float64, chunkSize int) {
	for i := 0; i < len(data); i += chunkSize {
		vec := data[i : i+chunkSize]

		max := math.Inf(-1)
		for _, x := range vec {
			if x > max {
				max = x
			}
		}

		var expSum float64
		for _, x := range vec {
			expSum += math.Exp(x - max)
		}
		sumLog := math.Log(expSum) + max

		for i, x := range vec {
			vec[i] = x - sumLog
		}
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

func applyUnitary(v Vector, f func(float64) float64) {
	data := v.Data()
	switch data := data.(type) {
	case []float32:
		for i, x := range data {
			data[i] = float32(f(float64(x)))
		}
	case []float64:
		for i, x := range data {
			data[i] = f(x)
		}
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
	v.SetData(data)
}

func applyAggregate(v Vector, f func([]float64) float64) Numeric {
	data := v.Data()
	switch data := data.(type) {
	case []float32:
		vec := make([]float64, len(data))
		for i, x := range data {
			vec[i] = float64(x)
		}
		return float32(f(vec))
	case []float64:
		return f(data)
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
}

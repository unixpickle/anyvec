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

// A Logger can take the natural logs of its entries.
type Logger interface {
	Log()
}

// Log takes the natural logarithm of the vector entries.
// If the vector does not implement Logger, a default
// implementation is used which supports float32 and
// float64 values.
func Log(v Vector) {
	if l, ok := v.(Logger); ok {
		l.Log()
	} else {
		applyUnitary(v, math.Log)
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

// A Power can raise its contents to a constant power.
type Power interface {
	Pow(p Numeric)
}

// Pow raises the vector to the n-th power.
// If the vector does not implement Power, a default
// implementation is used which supports float32 and
// float64 values.
func Pow(v Vector, n Numeric) {
	if p, ok := v.(Power); ok {
		p.Pow(n)
	} else {
		switch data := v.Data().(type) {
		case []float64:
			for i, x := range data {
				data[i] = math.Pow(x, n.(float64))
			}
			v.SetData(data)
		case []float32:
			for i, x := range data {
				data[i] = float32(math.Pow(float64(x), float64(n.(float32))))
			}
			v.SetData(data)
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
	}
}

// A LogAdder computes a sum of numbers which are stored
// in the log domain.
// It is equivalent to the pseudo-code:
//
//     sum = log(exp(v[0]) + exp(v[1]) + ...)
//
// Log addition is batched, meaning that the final result
// is list of sums, each from a different chunk of the
// input.
//
// The receiver's length must be divisible by the chunk
// size.
// If the chunk size is 0, then one addition is performed.
type LogAdder interface {
	AddLogs(chunkSize int) Vector
}

// AddLogs computes the sum of numbers which are in the
// log domain.
// If the vector does not implement LogAdder, a default
// implementation is used which supports float32 and
// float64 values.
func AddLogs(v Vector, chunkSize int) Vector {
	if l, ok := v.(LogAdder); ok {
		return l.AddLogs(chunkSize)
	} else {
		if chunkSize == 0 {
			chunkSize = v.Len()
		} else if v.Len()%chunkSize != 0 {
			panic("chunk size must divide vector size")
		}
		usableData := force64Bit(v.Data())
		var sum []float64
		for i := 0; i < len(usableData); i += chunkSize {
			sum = append(sum, logAdd(usableData[i:i+chunkSize]))
		}
		return v.Creator().MakeVectorData(v.Creator().MakeNumericList(sum))
	}
}

func logAdd(v []float64) float64 {
	if len(v) == 0 {
		return math.Inf(-1)
	}
	max := math.Inf(-1)
	for _, x := range v {
		if x > max {
			max = x
		}
	}
	var sum float64
	for _, x := range v {
		sum += math.Exp(x - max)
	}
	return math.Log(sum) + max
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

func force64Bit(slice interface{}) []float64 {
	switch data := slice.(type) {
	case []float64:
		return data
	case []float32:
		usableData := make([]float64, len(data))
		for i, x := range data {
			usableData[i] = float64(x)
		}
		return usableData
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
}

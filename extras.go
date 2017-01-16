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
// implementation is used.
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
// implementation is used.
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
// implementation is used.
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
// implementation is used.
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
// implementation is used.
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

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

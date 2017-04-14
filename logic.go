package anyvec

import "fmt"

// A Complementer can compute the probabilistic complement
// of its entries.
// In pseudo-code, this is "v[i] := 1 - v[i]".
type Complementer interface {
	Complement()
}

// Complement computes the probabilistic complement of the
// vector's entries.
// If the vector does not implement Complementer, a
// default implementation is used.
func Complement(v Vector) {
	if c, ok := v.(Complementer); ok {
		c.Complement()
	} else {
		v.Scale(v.Creator().MakeNumeric(-1))
		v.AddScalar(v.Creator().MakeNumeric(1))
	}
}

// A Comparer can compare its values to a constant,
// setting them to 1 if a condition is true or 0 if the
// condition is false.
//
// In some cases, a comparison might not make sense.
// For instance, imagine a vector full of complex numbers.
// In that case, it might be best to completely ignore the
// imaginary component while doing comparisons, so that
// comparisons behave as expected when the vectors are
// full of real numbers.
type Comparer interface {
	GreaterThan(n Numeric)
	LessThan(n Numeric)
	EqualTo(n Numeric)
}

// GreaterThan compares the components of a vector to a
// constant and sets the component to 1 if it is greater
// than the constant.
// If v doesn't implement Comparer, a default
// implementation is used which supports float32 and
// float64.
func GreaterThan(v Vector, n Numeric) {
	if c, ok := v.(Comparer); ok {
		c.GreaterThan(n)
	} else {
		applyComparison(v, n, func(x, y float32) bool {
			return x > y
		}, func(x, y float64) bool {
			return x > y
		})
	}
}

// LessThan is like GreaterThan, but it sets the entry to
// 1 if it is less than n.
func LessThan(v Vector, n Numeric) {
	if c, ok := v.(Comparer); ok {
		c.LessThan(n)
	} else {
		applyComparison(v, n, func(x, y float32) bool {
			return x < y
		}, func(x, y float64) bool {
			return x < y
		})
	}
}

// EqualTo is like GreaterThan or LessThan, but it sets
// the entry to 1 if it is equal to n.
func EqualTo(v Vector, n Numeric) {
	if c, ok := v.(Comparer); ok {
		c.EqualTo(n)
	} else {
		applyComparison(v, n, func(x, y float32) bool {
			return x == y
		}, func(x, y float64) bool {
			return x == y
		})
	}
}

func applyComparison(v Vector, n Numeric, c32 func(x, y float32) bool,
	c64 func(x, y float64) bool) {
	data := v.Data()
	switch data := data.(type) {
	case []float32:
		for i, x := range data {
			if c32(x, n.(float32)) {
				data[i] = 1
			} else {
				data[i] = 0
			}
		}
	case []float64:
		for i, x := range data {
			if c64(x, n.(float64)) {
				data[i] = 1
			} else {
				data[i] = 0
			}
		}
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
	v.SetData(data)
}

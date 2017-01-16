package anyvec

import "fmt"

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

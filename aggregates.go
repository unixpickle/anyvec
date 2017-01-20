package anyvec

import (
	"fmt"
	"math"

	"github.com/gonum/blas/blas32"
	"github.com/gonum/blas/blas64"
)

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

// An AbsSummer can sum up its entries' absolute values.
type AbsSummer interface {
	AbsSum() Numeric
}

// AbsSum sums the absolute values of the vector entries.
// If the vector does not implement AbsSummer, a default
// implementation is used which supports float32 and
// float64 values.
func AbsSum(v Vector) Numeric {
	if a, ok := v.(AbsSummer); ok {
		return a.AbsSum()
	} else {
		return applyAggregate(v, func(data []float64) float64 {
			var sum float64
			for _, x := range data {
				sum += math.Abs(x)
			}
			return sum
		})
	}
}

// An AbsMaxer can compute the greatest absolute value of
// its components.
type AbsMaxer interface {
	AbsMax() Numeric
}

// AbsMax finds the max of the absolute values of the
// vector entries.
// If the vector does not implement AbsMaxer, a default
// implementation is used which supports float32 and
// float64 values.
func AbsMax(v Vector) Numeric {
	if a, ok := v.(AbsMaxer); ok {
		return a.AbsMax()
	} else {
		return applyAggregate(v, func(data []float64) float64 {
			var max float64
			for _, x := range data {
				if m := math.Abs(x); m > max {
					max = m
				}
			}
			return max
		})
	}
}

// A Normer can compute the Euclidean norm of itself.
type Normer interface {
	Norm() Numeric
}

// Norm computes the Euclidean norm of the vector.
// If the vector does not implement Normer, a default
// implementation is used which supports float32 and
// float64 values.
func Norm(v Vector) Numeric {
	if n, ok := v.(Normer); ok {
		return n.Norm()
	} else {
		switch data := v.Data().(type) {
		case []float32:
			return blas32.Nrm2(v.Len(), blas32.Vector{Data: data, Inc: 1})
		case []float64:
			return blas64.Nrm2(v.Len(), blas64.Vector{Data: data, Inc: 1})
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
	}
}

// A MaxIndexer can find the index of its largest element.
type MaxIndexer interface {
	MaxIndex() int
}

// MaxIndex computes the index of the largest element.
// If the vector does not implement MaxIndexer, a default
// implementation is used which supports float32 and
// float64 values.
func MaxIndex(v Vector) int {
	if m, ok := v.(MaxIndexer); ok {
		return m.MaxIndex()
	} else {
		switch data := v.Data().(type) {
		case []float32:
			var idx int
			value := float32(math.Inf(-1))
			for i, x := range data {
				if x > value {
					value = x
					idx = i
				}
			}
			return idx
		case []float64:
			var idx int
			value := math.Inf(-1)
			for i, x := range data {
				if x > value {
					value = x
					idx = i
				}
			}
			return idx
		default:
			panic(fmt.Sprintf("unsupported type: %T", data))
		}
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

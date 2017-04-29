package anyvec64

import (
	"math"

	"github.com/unixpickle/anyvec"
)

// NumOps implements anyvec.NumOps for float64 numerics.
type NumOps struct{}

// Add adds numerics.
func (n NumOps) Add(n1, n2 anyvec.Numeric) anyvec.Numeric {
	return n1.(float64) + n2.(float64)
}

// Sub subtracts numerics.
func (n NumOps) Sub(n1, n2 anyvec.Numeric) anyvec.Numeric {
	return n1.(float64) - n2.(float64)
}

// Mul multiplies numerics.
func (n NumOps) Mul(n1, n2 anyvec.Numeric) anyvec.Numeric {
	return n1.(float64) * n2.(float64)
}

// Div divides numerics.
func (n NumOps) Div(n1, n2 anyvec.Numeric) anyvec.Numeric {
	return n1.(float64) / n2.(float64)
}

// Pow raises n1 to the n2 power.
func (n NumOps) Pow(n1, n2 anyvec.Numeric) anyvec.Numeric {
	return math.Pow(n1.(float64), n2.(float64))
}

// Identical is like Equal.
func (n NumOps) Identical(n1, n2 anyvec.Numeric) bool {
	return n.Equal(n1, n2)
}

// Equal checks if n1 == n2.
func (n NumOps) Equal(n1, n2 anyvec.Numeric) bool {
	return n1.(float64) == n2.(float64)
}

// Less checks if n1 < n2.
func (n NumOps) Less(n1, n2 anyvec.Numeric) bool {
	return n1.(float64) < n2.(float64)
}

// Greater checks if n1 > n2.
func (n NumOps) Greater(n1, n2 anyvec.Numeric) bool {
	return n1.(float64) > n2.(float64)
}

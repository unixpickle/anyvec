package anyvec

// NumOps performs elementary operations on Numerics.
//
// For all methods, the result is from (n1 "op" n2)
// where "op" is the operator described by the method
// name.
type NumOps interface {
	Add(n1, n2 Numeric) Numeric
	Sub(n1, n2 Numeric) Numeric
	Mul(n1, n2 Numeric) Numeric
	Div(n1, n2 Numeric) Numeric

	Equal(n1, n2 Numeric) bool

	// These may not work for certain types of
	// numerics such as complex numbers.
	Less(n1, n2 Numeric) bool
	Greater(n1, n2 Numeric) bool
}

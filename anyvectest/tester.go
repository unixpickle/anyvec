package anyvectest

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/anyvec"
)

const (
	singlePrec = 1e-3
	doublePrec = 1e-3
)

// Tester tests an anyvec.Creator which uses float32
// or float64 numerics.
type Tester struct {
	Creator anyvec.Creator
}

// TestAll runs every test.
func (t *Tester) TestAll(test *testing.T) {
	t.TestRequired(test)
	t.TestExtras(test)
}

// testBinOp tests a binary operation.
func (t *Tester) testBinOp(test *testing.T, op func(x, y float64) float64,
	doer func(v1, v2 anyvec.Vector)) {
	v1 := t.randomVec()
	v2 := t.randomVec()
	expected := t.unlist(v1.Data())
	for i, x := range t.unlist(v2.Data()) {
		expected[i] = op(expected[i], x)
	}
	lastV2 := v2.Data()
	doer(v1, v2)
	t.assertClose(test, v1.Data(), expected)
	t.assertClose(test, v2.Data(), lastV2)
}

// testOp tests a unary operation.
func (t *Tester) testOp(test *testing.T, op func(x float64) float64,
	doer func(v anyvec.Vector)) {
	v := t.randomVec()
	expected := make([]float64, v.Len())
	for i, x := range t.unlist(v.Data()) {
		expected[i] = op(x)
	}
	doer(v)
	t.assertClose(test, v.Data(), expected)
}

// testAgg tests an aggregate operation.
func (t *Tester) testAgg(test *testing.T, op func(x []float64) float64,
	doer func(v anyvec.Vector) anyvec.Numeric) {
	v := t.randomVec()
	expected := op(t.unlist(v.Data()))
	actual := doer(v)
	t.assertClose(test, actual, expected)
}

// is32Bit indicates whether or not the numeric types are
// 32-bit.
// This should only be necessary for functions which
// cannot use num/unnum/list/unlist for some reason.
func (t *Tester) is32Bit() bool {
	_, ok := t.Creator.MakeNumeric(1).(float32)
	return ok
}

// randomVec produces a random vector for testing.
// All random vectors will be of the same length.
func (t *Tester) randomVec() anyvec.Vector {
	return t.randomVecLen(8193)
}

// randomVecLen produces a random vector of a specified
// length.
func (t *Tester) randomVecLen(l int) anyvec.Vector {
	origVec := make([]float64, l)
	for i := range origVec {
		origVec[i] = rand.NormFloat64()
	}
	return t.vec(origVec)
}

// num produces a creator-compatible numeric.
func (t *Tester) num(x float64) anyvec.Numeric {
	return t.Creator.MakeNumeric(x)
}

// unnum produces a float64 for a numeric (either a
// float32 or a float64).
func (t *Tester) unnum(x anyvec.Numeric) float64 {
	switch x := x.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	default:
		panic(fmt.Sprintf("unsupported type: %T", x))
	}
}

// list produces a creator-compatible numeric list.
func (t *Tester) list(x []float64) anyvec.NumericList {
	return t.Creator.MakeNumericList(x)
}

// unlist produces a []float64 for a numeric list (either
// a []float32 or a []float64).
func (t *Tester) unlist(data anyvec.NumericList) []float64 {
	switch data := data.(type) {
	case []float64:
		return data
	case []float32:
		res := make([]float64, len(data))
		for i, x := range data {
			res[i] = float64(x)
		}
		return res
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
}

// vec produces a creator-compatible vector.
func (t *Tester) vec(x []float64) anyvec.Vector {
	return t.Creator.MakeVectorData(t.list(x))
}

// assertClose makes sure that two values or slices are
// close to each other within machine precision.
// Both actual and expected may be 32-bit or 64-bit bit,
// and they may eihter be slices or values.
// They do not have to be the same precision, but if one
// is a slice, so should be the other one.
func (t *Tester) assertClose(test *testing.T, actual, expected interface{}) {
	switch actual := actual.(type) {
	case float32:
		expected := t.unnum(expected)
		if !floatsClose(float64(actual), expected, singlePrec) {
			test.Errorf("expected %v but got %v", expected, actual)
		}
	case float64:
		expected := t.unnum(expected)
		if !floatsClose(actual, expected, doublePrec) {
			test.Errorf("expected %v but got %v", expected, actual)
		}
	case []float32:
		expected := t.unlist(expected)
		if len(actual) != len(expected) {
			test.Errorf("length mismatch")
			return
		}
		for i, x := range expected {
			y := actual[i]
			if !floatsClose(x, float64(y), singlePrec) {
				test.Errorf("index %d: expected %v but got %v", i, x, y)
				return
			}
		}
	case []float64:
		expected := t.unlist(expected)
		if len(actual) != len(expected) {
			test.Errorf("length mismatch")
			return
		}
		for i, x := range expected {
			y := actual[i]
			if !floatsClose(x, y, doublePrec) {
				test.Errorf("index %d: expected %v but got %v", i, x, y)
				return
			}
		}
	default:
		test.Fatalf("unsupported type: %T", actual)
	}
}

func floatsClose(f1, f2, prec float64) bool {
	if math.IsNaN(f1) {
		return math.IsNaN(f2)
	}
	return math.Abs(f1-f2) < prec
}

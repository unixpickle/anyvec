package anyvectest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
	"github.com/unixpickle/anyvec/anyvec32"
)

// Tester32 tests an anyvec32.Creator.
type Tester32 struct {
	Creator anyvec32.Creator
}

// TestAll runs every test.
func (t *Tester32) TestAll(test *testing.T) {
	t.TestRequired(test)
	t.TestExtras(test)
}

// TestRequired runs tests for all of the required Vector
// interface methods.
func (t *Tester32) TestRequired(test *testing.T) {
	test.Run("SliceConversion", t.TestSliceConversion)
	test.Run("Copy", t.TestCopy)
	test.Run("Scale", t.TestScale)
	test.Run("AddScaler", t.TestAddScaler)
	test.Run("Dot", t.TestDot)
	test.Run("Add", t.TestAdd)
	test.Run("Sub", t.TestSub)
	test.Run("Mul", t.TestMul)
	test.Run("Div", t.TestDiv)
	test.Run("Gemm", t.TestGemm)
}

// TestExtras runs tests for all of the possible extra
// methods the Vector may implement.
func (t *Tester32) TestExtras(test *testing.T) {
	test.Run("Exp", t.TestExp)
	test.Run("Sin", t.TestSin)
	test.Run("Tanh", t.TestTanh)
	test.Run("ClipPos", t.TestClipPos)
}

// TestSliceConversion makes sure that the vector properly
// translates to and from float32 values.
func (t *Tester32) TestSliceConversion(test *testing.T) {
	origVec := make([]float32, 513)
	for i := range origVec {
		origVec[i] = float32(rand.NormFloat64())
	}
	inVec := t.Creator.MakeVectorData(origVec)
	if inVec.Len() != len(origVec) {
		test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
	}
	if len(inVec.Data()) != len(origVec) {
		test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
	}
	origVec[0]++
	outData := inVec.Data()
	outData[0]--
	if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
		test.Error("invalid value after assignment")
	}
	if math.Abs(float64(origVec[0]-(inVec.Data()[0]+1))) > 1e-3 {
		test.Error("invalid value after assignment")
	}

	inVec.SetData(origVec)

	if inVec.Len() != len(origVec) {
		test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
	}
	if len(inVec.Data()) != len(origVec) {
		test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
	}

	if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
		test.Error("invalid value after assignment")
	}
	if math.Abs(float64(inVec.Data()[0]-(outData[0]+2))) > 1e-3 {
		test.Error("invalid value after assignment")
	}
	for i := 1; i < len(origVec); i++ {
		if math.Abs(float64(origVec[i]-inVec.Data()[i])) > 1e-3 {
			test.Errorf("bad value at index %d", i)
			break
		}
	}

	zeroVec := t.Creator.MakeVector(129)
	if zeroVec.Len() != 129 {
		test.Errorf("bad length: %d", zeroVec.Len())
	}
	for i, x := range zeroVec.Data() {
		if x != 0 {
			test.Errorf("should be 0 at index %d but got: %f", i, x)
			break
		}
	}
}

// TestCopy tests vector copying.
func (t *Tester32) TestCopy(test *testing.T) {
	origVec := make([]float32, 513)
	for i := range origVec {
		origVec[i] = float32(rand.NormFloat64())
	}

	vec1 := t.Creator.MakeVectorData(origVec)
	vec2 := vec1.Copy()

	origVec[37]++
	vec1.SetData(origVec)
	origVec[37] -= 2
	vec2.SetData(origVec)

	if math.Abs(float64(vec1.Data()[37]-(vec2.Data()[37]+2))) > 1e-3 {
		test.Error("values inconsistent after Copy()+SetData()")
	}
}

// TestScale tests vector scaling.
func (t *Tester32) TestScale(test *testing.T) {
	v := t.randomVec()
	data1 := v.Data()
	v.Scale(-0.5)
	data2 := v.Data()
	for i, x := range data1 {
		y := data2[i]
		if math.Abs(float64(x+2*y)) > 1e-3 || math.IsNaN(float64(x)) ||
			math.IsNaN(float64(y)) {
			test.Errorf("value at index %d: %v went to %v", i, x, y)
			return
		}
	}
}

// TestAddScaler tests scaler addition.
func (t *Tester32) TestAddScaler(test *testing.T) {
	v := t.randomVec()
	data1 := v.Data()
	v.AddScaler(-0.5)
	data2 := v.Data()
	for i, x := range data1 {
		y := data2[i]
		if math.Abs(float64(x-(y+0.5))) > 1e-3 || math.IsNaN(float64(x)) ||
			math.IsNaN(float64(y)) {
			test.Errorf("value at index %d: %v went to %v", i, x, y)
			return
		}
	}
}

// TestDot tests vector dot products.
func (t *Tester32) TestDot(test *testing.T) {
	v1 := t.randomVec()
	v2 := t.randomVec()
	var expected float32
	for i, x := range v1.Data() {
		expected += x * v2.Data()[i]
	}
	actual1 := v1.Dot(v2)
	actual2 := v2.Dot(v1)
	if math.Abs(float64(actual1-expected)) > 1e-3 {
		test.Errorf("expected %v but got %v", expected, actual1)
	}
	if math.Abs(float64(actual2-expected)) > 1e-3 {
		test.Errorf("expected %v but got %v", expected, actual2)
	}
}

// TestAdd tests vector addition.
func (t *Tester32) TestAdd(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x + y
	}, func(v1, v2 anyvec32.Vector) {
		v1.Add(v2)
	})
}

// TestSub tests vector subtraction.
func (t *Tester32) TestSub(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x - y
	}, func(v1, v2 anyvec32.Vector) {
		v1.Sub(v2)
	})
}

// TestMul tests vector multiplication.
func (t *Tester32) TestMul(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x * y
	}, func(v1, v2 anyvec32.Vector) {
		v1.Mul(v2)
	})
}

// TestDiv tests vector division.
func (t *Tester32) TestDiv(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x / y
	}, func(v1, v2 anyvec32.Vector) {
		v1.Div(v2)
	})
}

// TestGemm tests matrix multiplication.
func (t *Tester32) TestGemm(test *testing.T) {
	// TODO: fancier test for unusual strides.

	mat1 := &anyvec32.Matrix{
		Data: t.randomVecLen(30 * 17),
		Rows: 30,
		Cols: 17,
	}
	mat2 := &anyvec32.Matrix{
		Data: t.randomVecLen(30 * 5),
		Rows: 5,
		Cols: 30,
	}
	mat3 := &anyvec32.Matrix{
		Data: t.randomVecLen(30 * 5),
		Rows: 30,
		Cols: 5,
	}

	var bmats [3]blas32.General
	for i, x := range []*anyvec32.Matrix{mat1, mat2, mat3} {
		bmats[i] = blas32.General{
			Data:   x.Data.Data(),
			Rows:   x.Rows,
			Cols:   x.Cols,
			Stride: x.Cols,
		}
	}
	blas32.Gemm(blas.NoTrans, blas.Trans, 2.5, bmats[0], bmats[1], -0.7, bmats[2])
	mat3.Product(false, true, 2.5, mat1, mat2, -0.7)

	actual := mat3.Data.Data()
	expected := bmats[2].Data

	assertClose(test, actual, expected)
}

// TestExp test exponentiation.
func (t *Tester32) TestExp(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	}, anyvec32.Exp)
}

// TestSin test sine.
func (t *Tester32) TestSin(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Sin(float64(x)))
	}, anyvec32.Sin)
}

// TestTanh test tanh.
func (t *Tester32) TestTanh(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	}, anyvec32.Tanh)
}

// TestClipPos test positive clipping.
func (t *Tester32) TestClipPos(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Max(0, float64(x)))
	}, anyvec32.ClipPos)
}

// testBinOp tests a binary operation.
func (t *Tester32) testBinOp(test *testing.T, op func(x, y float32) float32,
	doer func(v1, v2 anyvec32.Vector)) {
	v1 := t.randomVec()
	v2 := t.randomVec()
	expected := v1.Data()
	for i, x := range v2.Data() {
		expected[i] = op(expected[i], x)
	}
	lastV2 := v2.Data()
	doer(v1, v2)
	assertClose(test, v1.Data(), expected)
	assertClose(test, v2.Data(), lastV2)
}

// testOp tests a unary operation.
func (t *Tester32) testOp(test *testing.T, op func(x float32) float32,
	doer func(v anyvec32.Vector)) {
	v := t.randomVec()
	expected := make([]float32, v.Len())
	for i, x := range v.Data() {
		expected[i] = op(x)
	}
	doer(v)
	assertClose(test, v.Data(), expected)
}

func (t *Tester32) randomVec() anyvec32.Vector {
	return t.randomVecLen(513)
}

func (t *Tester32) randomVecLen(l int) anyvec32.Vector {
	origVec := make([]float32, l)
	for i := range origVec {
		origVec[i] = float32(rand.NormFloat64())
	}
	return t.Creator.MakeVectorData(origVec)
}

func assertClose(t *testing.T, actual, expected []float32) {
	for i, x := range expected {
		y := actual[i]
		if math.Abs(float64(x-y)) > 1e-3 {
			t.Errorf("index %d: expected %v but got %v", i, x, y)
			return
		}
	}
}

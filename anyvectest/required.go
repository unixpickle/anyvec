package anyvectest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/anyvec"
)

// TestRequired runs tests for all of the required Vector
// interface methods.
func (t *Tester) TestRequired(test *testing.T) {
	test.Run("SliceConversion", t.TestSliceConversion)
	test.Run("Copy", t.TestCopy)
	test.Run("Set", t.TestSet)
	test.Run("Slice", t.TestSlice)
	test.Run("Concat", t.TestConcat)
	test.Run("Scale", t.TestScale)
	test.Run("AddScaler", t.TestAddScaler)
	test.Run("Dot", t.TestDot)
	test.Run("Add", t.TestAdd)
	test.Run("Sub", t.TestSub)
	test.Run("Mul", t.TestMul)
	test.Run("Div", t.TestDiv)
	test.Run("Gemm", t.TestGemm)
}

// TestSliceConversion makes sure that the vector properly
// translates to and from float values and that returned
// data slices are not the same slices backing the vector.
func (t *Tester) TestSliceConversion(test *testing.T) {
	// We cannot use t.unlist for this test because we
	// specifically want to see what happens if we write to
	// the original slice returned by Data().
	// Thus, it makes the most sense to have a separate
	// 32-bit and 64-bit test.
	if t.is32Bit() {
		origVec := make([]float32, 513)
		for i := range origVec {
			origVec[i] = float32(rand.NormFloat64())
		}
		inVec := t.Creator.MakeVectorData(origVec)
		if inVec.Len() != len(origVec) {
			test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
		}
		if len(inVec.Data().([]float32)) != len(origVec) {
			test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
		}
		origVec[0]++
		outData := inVec.Data().([]float32)
		outData[0]--
		if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		if math.Abs(float64(origVec[0]-(inVec.Data().([]float32)[0]+1))) > 1e-3 {
			test.Error("invalid value after assignment")
		}

		inVec.SetData(origVec)

		if inVec.Len() != len(origVec) {
			test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
		}
		if len(inVec.Data().([]float32)) != len(origVec) {
			test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
		}

		if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		if math.Abs(float64(inVec.Data().([]float32)[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		for i := 1; i < len(origVec); i++ {
			if math.Abs(float64(origVec[i]-inVec.Data().([]float32)[i])) > 1e-3 {
				test.Errorf("bad value at index %d", i)
				break
			}
		}
	} else {
		origVec := make([]float64, 513)
		for i := range origVec {
			origVec[i] = float64(rand.NormFloat64())
		}
		inVec := t.Creator.MakeVectorData(origVec)
		if inVec.Len() != len(origVec) {
			test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
		}
		if len(inVec.Data().([]float64)) != len(origVec) {
			test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
		}
		origVec[0]++
		outData := inVec.Data().([]float64)
		outData[0]--
		if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		if math.Abs(float64(origVec[0]-(inVec.Data().([]float64)[0]+1))) > 1e-3 {
			test.Error("invalid value after assignment")
		}

		inVec.SetData(origVec)

		if inVec.Len() != len(origVec) {
			test.Errorf("bad length: %d (expected %d)", inVec.Len(), len(origVec))
		}
		if len(inVec.Data().([]float64)) != len(origVec) {
			test.Errorf("bad len(Data()): %d (expected %d)", inVec.Len(), len(origVec))
		}

		if math.Abs(float64(origVec[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		if math.Abs(float64(inVec.Data().([]float64)[0]-(outData[0]+2))) > 1e-3 {
			test.Error("invalid value after assignment")
		}
		for i := 1; i < len(origVec); i++ {
			if math.Abs(float64(origVec[i]-inVec.Data().([]float64)[i])) > 1e-3 {
				test.Errorf("bad value at index %d", i)
				break
			}
		}
	}

	zeroVec := t.Creator.MakeVector(129)
	if zeroVec.Len() != 129 {
		test.Errorf("bad length: %d", zeroVec.Len())
	}
	for i, x := range t.unlist(zeroVec.Data()) {
		if x != 0 {
			test.Errorf("should be 0 at index %d but got: %f", i, x)
			break
		}
	}
}

// TestCopy tests vector copying.
func (t *Tester) TestCopy(test *testing.T) {
	// See TestSliceConversions for details on why we use
	// this if statement.
	if t.is32Bit() {
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

		if math.Abs(float64(vec1.Data().([]float32)[37]-(vec2.Data().([]float32)[37]+2))) > 1e-3 {
			test.Error("values inconsistent after Copy()+SetData()")
		}
	} else {
		origVec := make([]float64, 513)
		for i := range origVec {
			origVec[i] = float64(rand.NormFloat64())
		}

		vec1 := t.Creator.MakeVectorData(origVec)
		vec2 := vec1.Copy()

		origVec[37]++
		vec1.SetData(origVec)
		origVec[37] -= 2
		vec2.SetData(origVec)

		if math.Abs(float64(vec1.Data().([]float64)[37]-(vec2.Data().([]float64)[37]+2))) > 1e-3 {
			test.Error("values inconsistent after Copy()+SetData()")
		}
	}
}

// TestSet tests vector assignment.
func (t *Tester) TestSet(test *testing.T) {
	// See TestSliceConversions for details on why we use
	// this if statement.
	if t.is32Bit() {
		origVec := make([]float32, 513)
		for i := range origVec {
			origVec[i] = float32(rand.NormFloat64())
		}

		origVec[0] = 1.5

		vec1 := t.Creator.MakeVectorData(origVec)
		vec2 := t.Creator.MakeVector(len(origVec))
		old := vec2.Data().([]float32)
		vec2.Set(vec1)
		if old[0] != 0 {
			test.Errorf("expected 0 but got %f", old[0])
		}

		t.assertClose(test, vec1.Data().([]float32), origVec)
		t.assertClose(test, vec1.Data().([]float32), vec2.Data().([]float32))

		origVec[37]++
		vec1.SetData(origVec)
		origVec[37] -= 2
		vec2.SetData(origVec)

		if math.Abs(float64(vec1.Data().([]float32)[37]-(vec2.Data().([]float32)[37]+2))) > 1e-3 {
			test.Error("values inconsistent after Set()+SetData()")
		}
	} else {
		origVec := make([]float64, 513)
		for i := range origVec {
			origVec[i] = float64(rand.NormFloat64())
		}

		origVec[0] = 1.5

		vec1 := t.Creator.MakeVectorData(origVec)
		vec2 := t.Creator.MakeVector(len(origVec))
		old := vec2.Data().([]float64)
		vec2.Set(vec1)
		if old[0] != 0 {
			test.Errorf("expected 0 but got %f", old[0])
		}

		t.assertClose(test, vec1.Data().([]float64), origVec)
		t.assertClose(test, vec1.Data().([]float64), vec2.Data().([]float64))

		origVec[37]++
		vec1.SetData(origVec)
		origVec[37] -= 2
		vec2.SetData(origVec)

		if math.Abs(float64(vec1.Data().([]float64)[37]-(vec2.Data().([]float64)[37]+2))) > 1e-3 {
			test.Error("values inconsistent after Set()+SetData()")
		}
	}
}

// TestSlice tests vector slicing.
func (t *Tester) TestSlice(test *testing.T) {
	// See TestSliceConversions for details on why we use
	// this if statement.
	if t.is32Bit() {
		origVec := make([]float32, 513)
		for i := range origVec {
			origVec[i] = float32(rand.NormFloat64())
		}

		vec1 := t.Creator.MakeVectorData(origVec)
		vec2 := vec1.Slice(5, 20)

		actual := vec2.Data().([]float32)
		expected := origVec[5:20]

		t.assertClose(test, actual, expected)

		origVec[7] -= 10
		vec1.SetData(origVec)

		t.assertClose(test, vec2.Data().([]float32), actual)
	} else {
		origVec := make([]float64, 513)
		for i := range origVec {
			origVec[i] = float64(rand.NormFloat64())
		}

		vec1 := t.Creator.MakeVectorData(origVec)
		vec2 := vec1.Slice(5, 20)

		actual := vec2.Data().([]float64)
		expected := origVec[5:20]

		t.assertClose(test, actual, expected)

		origVec[7] -= 10
		vec1.SetData(origVec)

		t.assertClose(test, vec2.Data().([]float64), actual)
	}
}

// TestConcat tests vector concatenation.
func (t *Tester) TestConcat(test *testing.T) {
	// See TestSliceConversions for details on why we use
	// this if statement.
	if t.is32Bit() {
		data1 := make([]float32, 513)
		for i := range data1 {
			data1[i] = float32(rand.NormFloat64())
		}
		data2 := make([]float32, 13)
		for i := range data2 {
			data2[i] = float32(rand.NormFloat64())
		}
		data3 := make([]float32, 54)
		for i := range data2 {
			data2[i] = float32(rand.NormFloat64())
		}

		vec1 := t.Creator.MakeVectorData(data1)
		vec2 := t.Creator.MakeVectorData(data2)
		vec3 := t.Creator.MakeVectorData(data3)

		actual := t.Creator.Concat(vec1, vec2, vec3)
		expected := append(append(append([]float32{}, data1...), data2...), data3...)

		t.assertClose(test, actual.Data().([]float32), expected)
		old := actual.Data().([]float32)
		for i, x := range data1 {
			data1[i] = x - 1
		}
		t.assertClose(test, old, actual.Data().([]float32))
		vec1.SetData(data1)
		t.assertClose(test, old, actual.Data().([]float32))
	} else {
		data1 := make([]float64, 513)
		for i := range data1 {
			data1[i] = float64(rand.NormFloat64())
		}
		data2 := make([]float64, 13)
		for i := range data2 {
			data2[i] = float64(rand.NormFloat64())
		}
		data3 := make([]float64, 54)
		for i := range data2 {
			data2[i] = float64(rand.NormFloat64())
		}

		vec1 := t.Creator.MakeVectorData(data1)
		vec2 := t.Creator.MakeVectorData(data2)
		vec3 := t.Creator.MakeVectorData(data3)

		actual := t.Creator.Concat(vec1, vec2, vec3)
		expected := append(append(append([]float64{}, data1...), data2...), data3...)

		t.assertClose(test, actual.Data().([]float64), expected)
		old := actual.Data().([]float64)
		for i, x := range data1 {
			data1[i] = x - 1
		}
		t.assertClose(test, old, actual.Data().([]float64))
		vec1.SetData(data1)
		t.assertClose(test, old, actual.Data().([]float64))
	}
}

// TestScale tests vector scaling.
func (t *Tester) TestScale(test *testing.T) {
	v := t.randomVec()
	data1 := t.unlist(v.Data())
	v.Scale(t.num(-0.5))
	data2 := t.unlist(v.Data())
	for i, x := range data1 {
		y := data2[i]
		if math.Abs(x+2*y) > 1e-3 || math.IsNaN(x) || math.IsNaN(y) {
			test.Errorf("value at index %d: %v went to %v", i, x, y)
			return
		}
	}
}

// TestAddScaler tests scaler addition.
func (t *Tester) TestAddScaler(test *testing.T) {
	v := t.randomVec()
	data1 := t.unlist(v.Data())
	v.AddScaler(t.num(-0.5))
	data2 := t.unlist(v.Data())
	for i, x := range data1 {
		y := data2[i]
		if math.Abs(x-(y+0.5)) > 1e-3 || math.IsNaN(x) || math.IsNaN(y) {
			test.Errorf("value at index %d: %v went to %v", i, x, y)
			return
		}
	}
}

// TestDot tests vector dot products.
func (t *Tester) TestDot(test *testing.T) {
	v1 := t.randomVec()
	v2 := t.randomVec()
	var expected float64
	for i, x := range t.unlist(v1.Data()) {
		expected += x * t.unlist(v2.Data())[i]
	}
	actual1 := v1.Dot(v2)
	actual2 := v2.Dot(v1)
	t.assertClose(test, actual1, expected)
	t.assertClose(test, actual2, expected)
}

// TestAdd tests vector addition.
func (t *Tester) TestAdd(test *testing.T) {
	t.testBinOp(test, func(x, y float64) float64 {
		return x + y
	}, func(v1, v2 anyvec.Vector) {
		v1.Add(v2)
	})
}

// TestSub tests vector subtraction.
func (t *Tester) TestSub(test *testing.T) {
	t.testBinOp(test, func(x, y float64) float64 {
		return x - y
	}, func(v1, v2 anyvec.Vector) {
		v1.Sub(v2)
	})
}

// TestMul tests vector multiplication.
func (t *Tester) TestMul(test *testing.T) {
	t.testBinOp(test, func(x, y float64) float64 {
		return x * y
	}, func(v1, v2 anyvec.Vector) {
		v1.Mul(v2)
	})
}

// TestDiv tests vector division.
func (t *Tester) TestDiv(test *testing.T) {
	t.testBinOp(test, func(x, y float64) float64 {
		return x / y
	}, func(v1, v2 anyvec.Vector) {
		v1.Div(v2)
	})
}

// TestGemm tests matrix multiplication.
func (t *Tester) TestGemm(test *testing.T) {
	// TODO: fancier test for unusual strides.

	mat1 := &anyvec.Matrix{
		Data: t.randomVecLen(30 * 17),
		Rows: 30,
		Cols: 17,
	}
	mat2 := &anyvec.Matrix{
		Data: t.randomVecLen(30 * 5),
		Rows: 5,
		Cols: 30,
	}
	mat3 := &anyvec.Matrix{
		Data: t.randomVecLen(30 * 5),
		Rows: 30,
		Cols: 5,
	}

	var bmats [3]blas64.General
	for i, x := range []*anyvec.Matrix{mat1, mat2, mat3} {
		bmats[i] = blas64.General{
			Data:   t.unlist(x.Data.Data()),
			Rows:   x.Rows,
			Cols:   x.Cols,
			Stride: x.Cols,
		}
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 2.5, bmats[0], bmats[1], -0.7, bmats[2])
	mat3.Product(false, true, t.num(2.5), mat1, mat2, t.num(-0.7))

	actual := mat3.Data.Data()
	expected := bmats[2].Data

	t.assertClose(test, actual, expected)
}
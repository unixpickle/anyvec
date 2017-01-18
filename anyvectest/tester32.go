package anyvectest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas32"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/approb"
)

// Tester32 tests an anyvec.Creator which uses float32
// numerics.
type Tester32 struct {
	Creator anyvec.Creator
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

// TestExtras runs tests for all of the possible extra
// methods the Vector may implement.
func (t *Tester32) TestExtras(test *testing.T) {
	test.Run("Exp", t.TestExp)
	test.Run("Sin", t.TestSin)
	test.Run("Tanh", t.TestTanh)
	test.Run("ClipPos", t.TestClipPos)
	test.Run("Max", t.TestMax)
	test.Run("Sum", t.TestSum)
	test.Run("AbsMax", t.TestAbsMax)
	test.Run("AbsSum", t.TestAbsSum)
	test.Run("LogSoftmax", t.TestLogSoftmax)
	test.Run("ScaleChunks", t.TestScaleChunks)
	test.Run("Rand", t.TestRand)
	test.Run("AddRepeated", t.TestAddRepeated)
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

	zeroVec := t.Creator.MakeVector(129)
	if zeroVec.Len() != 129 {
		test.Errorf("bad length: %d", zeroVec.Len())
	}
	for i, x := range zeroVec.Data().([]float32) {
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

	if math.Abs(float64(vec1.Data().([]float32)[37]-(vec2.Data().([]float32)[37]+2))) > 1e-3 {
		test.Error("values inconsistent after Copy()+SetData()")
	}
}

// TestSet tests vector assignment.
func (t *Tester32) TestSet(test *testing.T) {
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

	assertClose(test, vec1.Data().([]float32), origVec)
	assertClose(test, vec1.Data().([]float32), vec2.Data().([]float32))

	origVec[37]++
	vec1.SetData(origVec)
	origVec[37] -= 2
	vec2.SetData(origVec)

	if math.Abs(float64(vec1.Data().([]float32)[37]-(vec2.Data().([]float32)[37]+2))) > 1e-3 {
		test.Error("values inconsistent after Set()+SetData()")
	}
}

// TestSlice tests vector slicing.
func (t *Tester32) TestSlice(test *testing.T) {
	origVec := make([]float32, 513)
	for i := range origVec {
		origVec[i] = float32(rand.NormFloat64())
	}

	vec1 := t.Creator.MakeVectorData(origVec)
	vec2 := vec1.Slice(5, 20)

	actual := vec2.Data().([]float32)
	expected := origVec[5:20]

	assertClose(test, actual, expected)

	origVec[7] -= 10
	vec1.SetData(origVec)

	assertClose(test, vec2.Data().([]float32), actual)
}

// TestConcat tests vector concatenation.
func (t *Tester32) TestConcat(test *testing.T) {
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

	assertClose(test, actual.Data().([]float32), expected)
	old := actual.Data().([]float32)
	for i, x := range data1 {
		data1[i] = x - 1
	}
	assertClose(test, old, actual.Data().([]float32))
	vec1.SetData(data1)
	assertClose(test, old, actual.Data().([]float32))
}

// TestScale tests vector scaling.
func (t *Tester32) TestScale(test *testing.T) {
	v := t.randomVec()
	data1 := v.Data().([]float32)
	v.Scale(float32(-0.5))
	data2 := v.Data().([]float32)
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
	data1 := v.Data().([]float32)
	v.AddScaler(float32(-0.5))
	data2 := v.Data().([]float32)
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
	for i, x := range v1.Data().([]float32) {
		expected += x * v2.Data().([]float32)[i]
	}
	actual1 := v1.Dot(v2).(float32)
	actual2 := v2.Dot(v1).(float32)
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
	}, func(v1, v2 anyvec.Vector) {
		v1.Add(v2)
	})
}

// TestSub tests vector subtraction.
func (t *Tester32) TestSub(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x - y
	}, func(v1, v2 anyvec.Vector) {
		v1.Sub(v2)
	})
}

// TestMul tests vector multiplication.
func (t *Tester32) TestMul(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x * y
	}, func(v1, v2 anyvec.Vector) {
		v1.Mul(v2)
	})
}

// TestDiv tests vector division.
func (t *Tester32) TestDiv(test *testing.T) {
	t.testBinOp(test, func(x, y float32) float32 {
		return x / y
	}, func(v1, v2 anyvec.Vector) {
		v1.Div(v2)
	})
}

// TestGemm tests matrix multiplication.
func (t *Tester32) TestGemm(test *testing.T) {
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

	var bmats [3]blas32.General
	for i, x := range []*anyvec.Matrix{mat1, mat2, mat3} {
		bmats[i] = blas32.General{
			Data:   x.Data.Data().([]float32),
			Rows:   x.Rows,
			Cols:   x.Cols,
			Stride: x.Cols,
		}
	}
	blas32.Gemm(blas.NoTrans, blas.Trans, 2.5, bmats[0], bmats[1], -0.7, bmats[2])
	mat3.Product(false, true, float32(2.5), mat1, mat2, float32(-0.7))

	actual := mat3.Data.Data().([]float32)
	expected := bmats[2].Data

	assertClose(test, actual, expected)
}

// TestExp tests exponentiation.
func (t *Tester32) TestExp(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	}, anyvec.Exp)
}

// TestSin tests sine.
func (t *Tester32) TestSin(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Sin(float64(x)))
	}, anyvec.Sin)
}

// TestTanh tests tanh.
func (t *Tester32) TestTanh(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	}, anyvec.Tanh)
}

// TestClipPos tests positive clipping.
func (t *Tester32) TestClipPos(test *testing.T) {
	t.testOp(test, func(x float32) float32 {
		return float32(math.Max(0, float64(x)))
	}, anyvec.ClipPos)
}

// TestSum test summation.
func (t *Tester32) TestSum(test *testing.T) {
	t.testAgg(test, func(x []float32) float32 {
		var sum float32
		for _, k := range x {
			sum += k
		}
		return sum
	}, anyvec.Sum)
}

// TestMax tests max computation.
func (t *Tester32) TestMax(test *testing.T) {
	t.testAgg(test, func(x []float32) float32 {
		max := float32(math.Inf(-1))
		for _, k := range x {
			if k > max {
				max = k
			}
		}
		return max
	}, anyvec.Max)
}

// TestAbsSum test absolute summation.
func (t *Tester32) TestAbsSum(test *testing.T) {
	t.testAgg(test, func(x []float32) float32 {
		var sum float32
		for _, k := range x {
			if k > 0 {
				sum += k
			} else {
				sum -= k
			}
		}
		return sum
	}, anyvec.AbsSum)
}

// TestAbsMax tests absolute max computation.
func (t *Tester32) TestAbsMax(test *testing.T) {
	t.testAgg(test, func(x []float32) float32 {
		var max float32
		for _, k := range x {
			x := k
			if x < 0 {
				x = -x
			}
			if x > max {
				max = x
			}
		}
		return max
	}, anyvec.AbsMax)
}

// TestLogSoftmax tests log-domain softmaxing.
func (t *Tester32) TestLogSoftmax(test *testing.T) {
	inData := []float32{0.735181, 0.638349, 0.672189, 0.025860, 0.492178,
		0.175069, 0.487960, 0.598212}
	expected := []float32{-1.84798499753082, -1.94481718265820, -1.91097680228496,
		-2.55730580715485, -2.09098781021549, -2.40809688666664, -2.09520595321559,
		-1.98495420701728}
	v := t.Creator.MakeVectorData(inData)
	anyvec.LogSoftmax(v, 0)
	assertClose(test, v.Data().([]float32), expected)

	expected = []float32{-1.20556282283073, -1.30239482283073, -1.26855482283073,
		-1.91488382283073, -1.34440459198132, -1.66151359198132, -1.34862259198132,
		-1.23837059198132}
	v = t.Creator.MakeVectorData(inData)
	anyvec.LogSoftmax(v, 4)
	assertClose(test, v.Data().([]float32), expected)

	inData = []float32{0.86568, -0.61489, 0.59658, -0.18710, -1.46797, -0.73938}
	expected = []float32{-0.97016, -2.45073, -1.23925, -2.02294, -3.30380, -2.57522}
	v = t.Creator.MakeVectorData(inData)
	anyvec.LogSoftmax(v, 6)
	assertClose(test, v.Data().([]float32), expected)
}

// TestScaleChunks tests chunk scaling.
func (t *Tester32) TestScaleChunks(test *testing.T) {
	data := []float32{-0.66886, -2.12313, -0.49031, -1.00323, -0.82617, 1.16384, -1.84009,
		-0.24389, 0.25384, -0.78570}
	scales := []float32{-0.25492, -0.50632}
	expected := []float32{0.17051, 0.54123, 0.12499, 0.25574, 0.21061, -0.58927, 0.93167,
		0.12349, -0.12852, 0.39781}
	v := t.Creator.MakeVectorData(data)
	anyvec.ScaleChunks(v, t.Creator.MakeVectorData(scales))
	assertClose(test, v.Data().([]float32), expected)
}

// TestRand tests random sampling.
func (t *Tester32) TestRand(test *testing.T) {
	const numSamples = 20000
	const vecSize = 15
	r := rand.New(rand.NewSource(1337))

	vec := t.Creator.MakeVector(vecSize)

	names := []string{"Uniform", "Bernoulli", "Normal"}
	dists := []anyvec.ProbDist{anyvec.Uniform, anyvec.Bernoulli, anyvec.Normal}
	samplers := []func() float64{rand.Float64, func() float64 {
		return float64(rand.Intn(2))
	}, rand.NormFloat64}
	for i, name := range names {
		actualSampler := func() float64 {
			if rand.Intn(2) == 0 {
				anyvec.Rand(vec, dists[i], r)
			} else {
				anyvec.Rand(vec, dists[i], nil)
			}
			return float64(vec.Data().([]float32)[rand.Intn(vecSize)])
		}
		corr := approb.Correlation(numSamples, 0.1, actualSampler, samplers[i])
		if corr < 0.95 {
			test.Errorf("distribution %s had correlation %f", name, corr)
		}
	}
}

// TestAddRepeated tests adding a repeated vector.
func (t *Tester32) TestAddRepeated(test *testing.T) {
	v := t.Creator.MakeVectorData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v1 := t.Creator.MakeVectorData([]float32{-3, -4, -5})

	anyvec.AddRepeated(v, v1)
	expected := []float32{1 - 3, 2 - 4, 3 - 5, 4 - 3, 5 - 4, 6 - 5, 7 - 3,
		8 - 4, 9 - 5, 10 - 3}
	assertClose(test, v.Data().([]float32), expected)

	anyvec.AddRepeated(v1, v)
	expected = []float32{-3 + 1 - 3, -4 + 2 - 4, -5 + 3 - 5}
	assertClose(test, v1.Data().([]float32), expected)

	v1 = t.Creator.MakeVectorData([]float32{-1, -2})
	v = t.Creator.MakeVectorData([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	anyvec.AddRepeated(v, v1)
	expected = []float32{0, 0, 3 - 1, 4 - 2, 5 - 1, 6 - 2, 7 - 1, 8 - 2}
	assertClose(test, v.Data().([]float32), expected)
}

// testBinOp tests a binary operation.
func (t *Tester32) testBinOp(test *testing.T, op func(x, y float32) float32,
	doer func(v1, v2 anyvec.Vector)) {
	v1 := t.randomVec()
	v2 := t.randomVec()
	expected := v1.Data().([]float32)
	for i, x := range v2.Data().([]float32) {
		expected[i] = op(expected[i], x)
	}
	lastV2 := v2.Data().([]float32)
	doer(v1, v2)
	assertClose(test, v1.Data().([]float32), expected)
	assertClose(test, v2.Data().([]float32), lastV2)
}

// testOp tests a unary operation.
func (t *Tester32) testOp(test *testing.T, op func(x float32) float32,
	doer func(v anyvec.Vector)) {
	v := t.randomVec()
	expected := make([]float32, v.Len())
	for i, x := range v.Data().([]float32) {
		expected[i] = op(x)
	}
	doer(v)
	assertClose(test, v.Data().([]float32), expected)
}

// testAgg tests an aggregate operation.
func (t *Tester32) testAgg(test *testing.T, op func(x []float32) float32,
	doer func(v anyvec.Vector) anyvec.Numeric) {
	v := t.randomVec()
	expected := op(v.Data().([]float32))
	actual := doer(v).(float32)
	if math.Abs(float64(actual-expected)) > 1e-3 {
		test.Errorf("expected %v but got %v", expected, actual)
	}
}

func (t *Tester32) randomVec() anyvec.Vector {
	return t.randomVecLen(8192)
}

func (t *Tester32) randomVecLen(l int) anyvec.Vector {
	origVec := make([]float32, l)
	for i := range origVec {
		origVec[i] = float32(rand.NormFloat64())
	}
	return t.Creator.MakeVectorData(origVec)
}

func assertClose(t *testing.T, actual, expected []float32) {
	if len(actual) != len(expected) {
		t.Errorf("length mismatch")
		return
	}
	for i, x := range expected {
		y := actual[i]
		if math.Abs(float64(x-y)) > 1e-3 {
			t.Errorf("index %d: expected %v but got %v", i, x, y)
			return
		}
	}
}

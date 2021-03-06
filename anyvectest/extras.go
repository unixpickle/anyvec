package anyvectest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/approb"
	"github.com/unixpickle/essentials"
)

// TestExtras runs tests for all of the possible extra
// methods the Vector may implement.
func (t *Tester) TestExtras(test *testing.T) {
	test.Run("Exp", t.TestExp)
	test.Run("Log", t.TestLog)
	test.Run("Sin", t.TestSin)
	test.Run("Cos", t.TestCos)
	test.Run("Sigmoid", t.TestSigmoid)
	test.Run("Tanh", t.TestTanh)
	test.Run("ClipPos", t.TestClipPos)
	test.Run("ClipRange", t.TestClipRange)
	test.Run("Round", t.TestRound)
	test.Run("Pow", t.TestPow)
	test.Run("Max", t.TestMax)
	test.Run("MaxIndex", t.TestMaxIndex)
	test.Run("Sum", t.TestSum)
	test.Run("AbsMax", t.TestAbsMax)
	test.Run("AbsSum", t.TestAbsSum)
	test.Run("Norm", t.TestNorm)
	test.Run("ElemMax", t.TestElemMax)
	test.Run("LogSoftmax", t.TestLogSoftmax)
	test.Run("AddChunks", t.TestAddChunks)
	test.Run("ScaleChunks", t.TestScaleChunks)
	test.Run("Rand", t.TestRand)
	test.Run("AddRepeated", t.TestAddRepeated)
	test.Run("ScaleRepeated", t.TestScaleRepeated)
	test.Run("Comparisons", t.TestComparisons)
	test.Run("Complement", t.TestComplement)
	test.Run("AddLogs", t.TestAddLogs)
	test.Run("SumRows", t.TestSumRows)
	test.Run("SumCols", t.TestSumCols)
	test.Run("MapMax", t.TestMapMax)
	test.Run("Transpose", t.TestTranspose)
	test.Run("Gemv", t.TestGemv)
	test.Run("Gemm", t.TestGemm)
	test.Run("BatchedGemm", t.TestBatchedGemm)
}

// TestExp tests exponentiation.
func (t *Tester) TestExp(test *testing.T) {
	t.testOp(test, math.Exp, anyvec.Exp)
}

// TestLog tests logarithms.
func (t *Tester) TestLog(test *testing.T) {
	t.testOp(test, math.Log, anyvec.Log)
}

// TestSin tests sine.
func (t *Tester) TestSin(test *testing.T) {
	t.testOp(test, math.Sin, anyvec.Sin)
}

// TestCos tests cosine.
func (t *Tester) TestCos(test *testing.T) {
	t.testOp(test, math.Cos, anyvec.Cos)
}

// TestSin tests logistic sigmoid.
func (t *Tester) TestSigmoid(test *testing.T) {
	t.testOp(test, func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}, anyvec.Sigmoid)
}

// TestTanh tests tanh.
func (t *Tester) TestTanh(test *testing.T) {
	t.testOp(test, math.Tanh, anyvec.Tanh)
}

// TestClipPos tests positive clipping.
func (t *Tester) TestClipPos(test *testing.T) {
	t.testOp(test, func(x float64) float64 {
		return math.Max(0, x)
	}, anyvec.ClipPos)
}

// TestClipRange tests arbitrary range clipping.
func (t *Tester) TestClipRange(test *testing.T) {
	mins := []float64{-0.3, 0.2, -100}
	maxes := []float64{0.9, 1.5, 100}
	for i, min := range mins {
		max := maxes[i]
		t.testOp(test, func(x float64) float64 {
			return math.Min(math.Max(x, min), max)
		}, func(in anyvec.Vector) {
			c := in.Creator()
			anyvec.ClipRange(in, c.MakeNumeric(min), c.MakeNumeric(max))
		})
	}
}

// TestRound tests rounding.
func (t *Tester) TestRound(test *testing.T) {
	t.testOp(test, essentials.Round, anyvec.Round)
}

// TestPow tests power taking.
func (t *Tester) TestPow(test *testing.T) {
	t.testOp(test, func(x float64) float64 {
		return math.Pow(x/5, 2)
	}, func(v anyvec.Vector) {
		v.Scale(t.num(0.2))
		anyvec.Pow(v, t.num(2))
	})

	for _, negPower := range []float64{-2.5, -0.5, -2} {
		data := make([]float64, 10)
		expected := make([]float64, 10)
		for i := range data {
			data[i] = math.Abs(rand.NormFloat64()) + 1
			expected[i] = math.Pow(data[i], negPower)
		}
		v := t.vec(data)
		anyvec.Pow(v, t.num(negPower))
		t.assertClose(test, v.Data(), expected)
	}

	for _, fracPower := range []float64{0.3, 0.5} {
		v := t.vec([]float64{rand.Float64(), rand.Float64() * 10, rand.Float64() * 100})
		old := v.Data()
		anyvec.Pow(v, t.num(fracPower))
		anyvec.Pow(v, t.num(1/fracPower))
		t.assertClose(test, v.Data(), old)
	}
}

// TestSum tests summation.
func (t *Tester) TestSum(test *testing.T) {
	t.testAgg(test, func(x []float64) float64 {
		var sum float64
		for _, k := range x {
			sum += k
		}
		return sum
	}, anyvec.Sum)
}

// TestMax tests max computation.
func (t *Tester) TestMax(test *testing.T) {
	t.testAgg(test, func(x []float64) float64 {
		max := math.Inf(-1)
		for _, k := range x {
			if k > max {
				max = k
			}
		}
		return max
	}, anyvec.Max)
}

// TestMaxIndex tests max index computation.
func (t *Tester) TestMaxIndex(test *testing.T) {
	// If each random vector is 512, then the probability of
	// getting the same index 3 times in a row is 2^(-36).
	for i := 0; i < 4; i++ {
		vec := t.randomVecLen(512)
		max := math.Inf(-1)
		maxIdx := 0
		for i, x := range t.unlist(vec.Data()) {
			if x > max {
				max = x
				maxIdx = i
			}
		}
		actual := anyvec.MaxIndex(vec)
		if maxIdx != actual {
			test.Errorf("expected index %d but got %d", maxIdx, actual)
		}
	}
}

// TestAbsSum tests absolute summation.
func (t *Tester) TestAbsSum(test *testing.T) {
	t.testAgg(test, func(x []float64) float64 {
		var sum float64
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

// TestNorm tests the Euclidean norm.
func (t *Tester) TestNorm(test *testing.T) {
	t.testAgg(test, func(x []float64) float64 {
		var sum float64
		for _, k := range x {
			sum += k * k
		}
		return math.Sqrt(sum)
	}, anyvec.Norm)
}

// TestAbsMax tests absolute max computation.
func (t *Tester) TestAbsMax(test *testing.T) {
	t.testAgg(test, func(x []float64) float64 {
		var max float64
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

// TestElemMax tests the ElemMaxer interface.
func (t *Tester) TestElemMax(test *testing.T) {
	inData1 := []float64{1, -2, 3, -4, 5, 6, 3, 1.5}
	inData2 := []float64{-1, -1, 3.5, -5, -6, 6, 2.5, 2}
	v1 := t.vec(inData1)
	v2 := t.vec(inData2)
	anyvec.ElemMax(v1, v2)

	expected := []float64{1, -1, 3.5, -4, 5, 6, 3, 2}
	t.assertClose(test, v1.Data(), expected)
}

// TestLogSoftmax tests log-domain softmaxing.
func (t *Tester) TestLogSoftmax(test *testing.T) {
	inData := []float64{0.735181, 0.638349, 0.672189, 0.025860, 0.492178,
		0.175069, 0.487960, 0.598212}
	expected := []float64{-1.84798499753082, -1.94481718265820, -1.91097680228496,
		-2.55730580715485, -2.09098781021549, -2.40809688666664, -2.09520595321559,
		-1.98495420701728}
	v := t.vec(inData)
	anyvec.LogSoftmax(v, 0)
	t.assertClose(test, v.Data(), expected)

	expected = []float64{-1.20556282283073, -1.30239482283073, -1.26855482283073,
		-1.91488382283073, -1.34440459198132, -1.66151359198132, -1.34862259198132,
		-1.23837059198132}
	v = t.vec(inData)
	anyvec.LogSoftmax(v, 4)
	t.assertClose(test, v.Data(), expected)

	inData = []float64{0.86568, -0.61489, 0.59658, -0.18710, -1.46797, -0.73938}
	expected = []float64{-0.97016, -2.45073, -1.23925, -2.02294, -3.30380, -2.57522}
	v = t.vec(inData)
	anyvec.LogSoftmax(v, 6)
	t.assertClose(test, v.Data(), expected)
}

// TestAddChunks tests chunk addition.
func (t *Tester) TestAddChunks(test *testing.T) {
	data := []float64{-0.66886, -2.12313, -0.49031, -1.00323, -0.82617, 1.16384, -1.84009,
		-0.24389, 0.25384, -0.78570}
	scalers := []float64{-0.25492, -0.50632}
	expected := make([]float64, len(data))
	for i, x := range data {
		expected[i] = x + scalers[i/5]
	}
	v := t.vec(data)
	anyvec.AddChunks(v, t.vec(scalers))
	t.assertClose(test, v.Data(), expected)
}

// TestScaleChunks tests chunk scaling.
func (t *Tester) TestScaleChunks(test *testing.T) {
	data := []float64{-0.66886, -2.12313, -0.49031, -1.00323, -0.82617, 1.16384, -1.84009,
		-0.24389, 0.25384, -0.78570}
	scales := []float64{-0.25492, -0.50632}
	expected := []float64{0.17051, 0.54123, 0.12499, 0.25574, 0.21061, -0.58927, 0.93167,
		0.12349, -0.12852, 0.39781}
	v := t.vec(data)
	anyvec.ScaleChunks(v, t.vec(scales))
	t.assertClose(test, v.Data(), expected)
}

// TestRand tests random sampling.
func (t *Tester) TestRand(test *testing.T) {
	if testing.Short() {
		test.Skip("not running in short mode")
	}

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
			return t.unlist(vec.Data())[rand.Intn(vecSize)]
		}
		corr := approb.Correlation(numSamples, 0.1, actualSampler, samplers[i])
		if corr < 0.95 {
			test.Errorf("distribution %s had correlation %f", name, corr)
		}
	}
}

// TestAddRepeated tests adding a repeated vector.
func (t *Tester) TestAddRepeated(test *testing.T) {
	v := t.vec([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v1 := t.vec([]float64{-3, -4, -5})

	anyvec.AddRepeated(v, v1)
	expected := []float64{1 - 3, 2 - 4, 3 - 5, 4 - 3, 5 - 4, 6 - 5, 7 - 3,
		8 - 4, 9 - 5, 10 - 3}
	t.assertClose(test, v.Data(), expected)

	anyvec.AddRepeated(v1, v)
	expected = []float64{-3 + 1 - 3, -4 + 2 - 4, -5 + 3 - 5}
	t.assertClose(test, v1.Data(), expected)

	v1 = t.vec([]float64{-1, -2})
	v = t.vec([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	anyvec.AddRepeated(v, v1)
	expected = []float64{0, 0, 3 - 1, 4 - 2, 5 - 1, 6 - 2, 7 - 1, 8 - 2}
	t.assertClose(test, v.Data(), expected)
}

// TestScaleRepeated tests scaling a repeated vector.
func (t *Tester) TestScaleRepeated(test *testing.T) {
	v := t.vec([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v1 := t.vec([]float64{-3, -4, -5})

	anyvec.ScaleRepeated(v, v1)
	expected := []float64{1 * -3, 2 * -4, 3 * -5, 4 * -3, 5 * -4, 6 * -5, 7 * -3,
		8 * -4, 9 * -5, 10 * -3}
	t.assertClose(test, v.Data(), expected)

	anyvec.ScaleRepeated(v1, v)
	expected = []float64{-3 * expected[0], -4 * expected[1], -5 * expected[2]}
	t.assertClose(test, v1.Data(), expected)

	v1 = t.vec([]float64{-1, -2})
	v = t.vec([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	anyvec.ScaleRepeated(v, v1)
	expected = []float64{-1, -4, -3, -8, -5, -12, -7, -16}
	t.assertClose(test, v.Data(), expected)
}

// TestComparisons tests the Comparer interface.
func (t *Tester) TestComparisons(test *testing.T) {
	data := []float64{1, -1, 2, -2, 3, -3, 1.5, 7, 0, 1}
	v := t.vec(data)
	anyvec.GreaterThan(v, t.num(1.5))
	expected := []float64{0, 0, 1, 0, 1, 0, 0, 1, 0, 0}
	t.assertClose(test, v.Data(), expected)

	v = t.vec(data)
	anyvec.LessThan(v, t.num(-1))
	expected = []float64{0, 0, 0, 1, 0, 1, 0, 0, 0, 0}
	t.assertClose(test, v.Data(), expected)

	v = t.vec(data)
	anyvec.EqualTo(v, t.num(2))
	expected = []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	t.assertClose(test, v.Data(), expected)
}

// TestComplement tests the Complementer interface.
func (t *Tester) TestComplement(test *testing.T) {
	data := []float64{1, -1, 2, -2, 3, -3, 1.5, 7, 0, 1}
	v := t.vec(data)
	anyvec.Complement(v)
	expected := []float64{0, 2, -1, 3, -2, 4, -0.5, -6, 1, 0}
	t.assertClose(test, v.Data(), expected)
}

// TestAddLogs tests the LogAdder interface.
func (t *Tester) TestAddLogs(test *testing.T) {
	data := []float64{1, 2, 0.5, -2.5, 1, -1, -1, -0.5}
	v := t.vec(data)
	sum := anyvec.AddLogs(v, 4)
	expected := []float64{2.47132687702842, 1.40132369536570}
	t.assertClose(test, sum.Data(), expected)
	sum = anyvec.AddLogs(v, 0)
	expected = []float64{2.766238324}
	t.assertClose(test, sum.Data(), expected)

	if !testing.Short() {
		data = make([]float64, 300*3)
		expected = make([]float64, len(data)/300)
		for i := range data {
			data[i] = rand.NormFloat64()
			expected[i/300] += math.Exp(data[i])
		}
		for i, x := range expected {
			expected[i] = math.Log(x)
		}
		v = t.vec(data)
		sum = anyvec.AddLogs(v, 300)
		t.assertClose(test, sum.Data(), expected)

		data := make([]float64, 65537*2)
		expected := make([]float64, len(data)/2)
		for i := range data {
			data[i] = rand.NormFloat64()
			expected[i/2] += math.Exp(data[i])
		}
		for i, x := range expected {
			expected[i] = math.Log(x)
		}
		v = t.vec(data)
		sum = anyvec.AddLogs(v, 2)
		t.assertClose(test, sum.Data(), expected)
	}
}

// TestSumRows tests the RowSummer interface.
func (t *Tester) TestSumRows(test *testing.T) {
	v := t.vec([]float64{
		0.305985279638971, 1.058956965872622, -0.780664183736345,
		-2.213119641375825, 0.511557156875536, -1.025825432279111,
		-0.193978187968917, 1.247563683930367, 0.769710039307595,
		0.699426864914262, 1.165449702401210, 0.500689376716200,
	})
	actual := anyvec.SumRows(v, 3).Data()
	expected := []float64{-1.401685684791509, 3.983527509079735, -0.536090199991662}
	t.assertClose(test, actual, expected)
}

// TestSumCols tests the ColSummer interface.
func (t *Tester) TestSumCols(test *testing.T) {
	v := t.vec([]float64{
		0.305985279638971, 1.058956965872622, -0.780664183736345,
		-2.213119641375825, 0.511557156875536, -1.025825432279111,
		-0.193978187968917, 1.247563683930367, 0.769710039307595,
		0.699426864914262, 1.165449702401210, 0.500689376716200,
	})
	actual := anyvec.SumCols(v, 4).Data()
	expected := []float64{0.584278061775247, -2.727387916779400, 1.823295535269045,
		2.365565944031672}
	t.assertClose(test, actual, expected)
}

// TestMapMax tests the MaxMapper interface.
func (t *Tester) TestMapMax(test *testing.T) {
	v := t.vec([]float64{
		0.305985279638971, 1.058956965872622, -0.780664183736345,
		-2.213119641375825, 0.511557156875536, -1.025825432279111,
		-0.193978187968917, 0.769710039307595, 1.247563683930367,
		1.165449702401210, 0.699426864914262, 0.500689376716200,
	})
	mapper := anyvec.MapMax(v, 3)
	mapMe := t.vec([]float64{
		-1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	actual := t.Creator.MakeVector(4)
	mapper.Map(mapMe, actual)
	expected := []float64{2, 5, 9, 10}
	t.assertClose(test, actual.Data(), expected)

	mapper = anyvec.MapMax(v, 4)
	mapMe = t.vec([]float64{
		-1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	actual = t.Creator.MakeVector(3)
	mapper.Map(mapMe, actual)
	expected = []float64{2, 8, 9}
	t.assertClose(test, actual.Data(), expected)
}

// TestTranspose tests the Transposer interface.
func (t *Tester) TestTranspose(test *testing.T) {
	v := t.vec([]float64{
		0.305985279638971, 1.058956965872622, -0.780664183736345,
		-2.213119641375825, 0.511557156875536, -1.025825432279111,
		-0.193978187968917, 0.769710039307595, 1.247563683930367,
		1.165449702401210, 0.699426864914262, 0.500689376716200,
	})
	in := &anyvec.Matrix{Data: v, Rows: 4, Cols: 3}
	out := &anyvec.Matrix{Data: t.Creator.MakeVector(3 * 4), Rows: 3, Cols: 4}
	out.Transpose(in)

	expected := []float64{
		0.305985279638971, -2.213119641375825, -0.193978187968917, 1.165449702401210,
		1.058956965872622, 0.511557156875536, 0.769710039307595, 0.699426864914262,
		-0.780664183736345, -1.025825432279111, 1.247563683930367, 0.500689376716200,
	}

	t.assertClose(test, out.Data.Data(), expected)
}

// TestGemv tests the Gemver interface.
func (t *Tester) TestGemv(test *testing.T) {
	v := t.vec([]float64{
		0.305985279638971, 1.058956965872622, -0.780664183736345,
		-2.213119641375825, 0.511557156875536, -1.025825432279111,
		-0.193978187968917, 0.769710039307595, 1.247563683930367,
		1.165449702401210, 0.699426864914262, 0.500689376716200,
	})
	v1 := t.vec([]float64{
		3.5, 7, 2.7,
	})
	product := t.vec([]float64{1, 7, 3, 5, -1, 3, 2, 9})

	anyvec.Gemv(false, 4, 2, t.num(1), v, 3, v1, 2, t.num(0.5), product, 2)

	expected := []float64{
		4.430132286592478, 7,
		-4.864714421251440, 5,
		0.899293448239297, 3,
		6.967526493672743, 9,
	}

	t.assertClose(test, product.Data(), expected)
}

// TestGemm tests the Gemmer interface.
func (t *Tester) TestGemm(test *testing.T) {
	// TODO: fancier test for unusual strides.

	mat1 := &anyvec.Matrix{
		Data: t.randomVecLen(30 * 17),
		Rows: 30,
		Cols: 17,
	}
	mat2 := &anyvec.Matrix{
		Data: t.randomVecLen(17 * 5),
		Rows: 5,
		Cols: 17,
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

// TestBatchedGemm tests the BatchedGemmer interface.
func (t *Tester) TestBatchedGemm(test *testing.T) {
	const batchSize = 7

	mat1 := &anyvec.MatrixBatch{
		Data: t.randomVecLen(30 * 17 * batchSize),
		Rows: 30,
		Cols: 17,
		Num:  batchSize,
	}
	mat2 := &anyvec.MatrixBatch{
		Data: t.randomVecLen(17 * 5 * batchSize),
		Rows: 5,
		Cols: 17,
		Num:  batchSize,
	}
	mat3 := &anyvec.MatrixBatch{
		Data: t.randomVecLen(30 * 5 * batchSize),
		Rows: 30,
		Cols: 5,
		Num:  batchSize,
	}

	var expected []float64

	for j := 0; j < batchSize; j++ {
		var bmats [3]blas64.General
		for i, x := range []*anyvec.MatrixBatch{mat1, mat2, mat3} {
			list := t.unlist(x.Data.Data())
			chunkSize := len(list) / batchSize
			bmats[i] = blas64.General{
				Data:   list[chunkSize*j : chunkSize*(j+1)],
				Rows:   x.Rows,
				Cols:   x.Cols,
				Stride: x.Cols,
			}
		}
		blas64.Gemm(blas.NoTrans, blas.Trans, 2.5, bmats[0], bmats[1], -0.7, bmats[2])
		expected = append(expected, bmats[2].Data...)
	}

	mat3.Product(false, true, t.num(2.5), mat1, mat2, t.num(-0.7))
	actual := mat3.Data.Data()
	t.assertClose(test, actual, expected)
}

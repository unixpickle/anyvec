package cuda

import (
	"math/rand"
	"runtime"
	"unsafe"

	"github.com/unixpickle/anyvec"
)

// A Creator32 implements anyvec.Creator for float32
// numerics.
type Creator32 struct {
	handle *Handle
}

// NewCreator32 creates a Creator32 that uses a given CUDA
// handle.
func NewCreator32(h *Handle) *Creator32 {
	return &Creator32{handle: h}
}

// MakeNumeric creates a float32.
func (c *Creator32) MakeNumeric(x float64) anyvec.Numeric {
	return float32(x)
}

// MakeNumericList creates a []float32.
func (c *Creator32) MakeNumericList(x []float64) anyvec.NumericList {
	res := make([]float32, len(x))
	for i, k := range x {
		res[i] = float32(k)
	}
	return res
}

// MakeVector creates a zero'd out anyvec.Vector.
func (c *Creator32) MakeVector(size int) anyvec.Vector {
	buf, err := newBuffer(c.handle, size*4)
	if err != nil {
		panic(err)
	}
	if err := buf.Clear(); err != nil {
		panic(err)
	}
	return &vector32{
		creator: c,
		buffer:  buf,
	}
}

// MakeVectorData creates an anyvec.Vector with the
// specified contents.
func (c *Creator32) MakeVectorData(dObj anyvec.NumericList) anyvec.Vector {
	d := dObj.([]float32)
	buf, err := newBuffer(c.handle, len(d)*4)
	if err != nil {
		panic(err)
	}
	if err := buf.Set32(d); err != nil {
		panic(err)
	}
	return &vector32{
		creator: c,
		buffer:  buf,
	}
}

// Concat concatenates vectors.
func (c *Creator32) Concat(v ...anyvec.Vector) anyvec.Vector {
	bufs := make([]*buffer, len(v))
	for i, x := range v {
		bufs[i] = x.(*vector32).buffer
	}
	buf, err := newBufferConcat(c.handle, bufs)
	if err != nil {
		panic(err)
	}
	return &vector32{
		creator: c,
		buffer:  buf,
	}
}

type vector32 struct {
	creator *Creator32
	buffer  *buffer
}

func (v *vector32) Creator() anyvec.Creator {
	return v.creator
}

func (v *vector32) Len() int {
	return v.buffer.Len() / 4
}

func (v *vector32) Data() anyvec.NumericList {
	res := make([]float32, v.Len())
	if err := v.buffer.Get32(res); err != nil {
		panic(err)
	}
	return res
}

func (v *vector32) SetData(d anyvec.NumericList) {
	if err := v.buffer.Set32(d.([]float32)); err != nil {
		panic(err)
	}
}

func (v *vector32) Set(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	v.buffer.Set(v1.(*vector32).buffer)
}

func (v *vector32) Copy() anyvec.Vector {
	newBuff, err := newBuffer(v.creator.handle, v.buffer.size)
	if err != nil {
		panic(err)
	}
	if err := newBuff.Set(v.buffer); err != nil {
		panic(err)
	}
	return &vector32{
		creator: v.creator,
		buffer:  newBuff,
	}
}

func (v *vector32) Slice(start, end int) anyvec.Vector {
	if start < 0 || end < 0 {
		panic("indices must be non-negative")
	}
	if end < start {
		panic("invalid range: end < start")
	}
	if end > v.Len() {
		panic("end out of bounds")
	}
	buf, err := newBuffer(v.creator.handle, (end-start)*4)
	if err != nil {
		panic(err)
	}
	buf.Set(&buffer{
		size: (end - start) * 4,
		ptr:  unsafe.Pointer(uintptr(v.buffer.ptr) + uintptr(4*start)),
	})
	runtime.KeepAlive(v.buffer)
	return &vector32{
		creator: v.creator,
		buffer:  buf,
	}
}

func (v *vector32) Scale(s anyvec.Numeric) {
	v.ops().Scal(v.Len(), s.(float32), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) AddScaler(s anyvec.Numeric) {
	v.ops().AddScaler(v.Len(), s.(float32), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) Dot(v1 anyvec.Vector) anyvec.Numeric {
	v.assertMatch(v1)
	res := v.ops().Dot(v.Len(), v.buffer.ptr, v1.(*vector32).buffer.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1)
	return res
}

func (v *vector32) Add(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	v.assertMatch(v1)
	v.ops().Axpy(v.Len(), 1, v1.(*vector32).buffer.ptr, v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1)
}

func (v *vector32) Sub(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	v.assertMatch(v1)
	v.ops().Axpy(v.Len(), -1, v1.(*vector32).buffer.ptr, v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1)
}

func (v *vector32) Mul(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	v.assertMatch(v1)
	v.ops().Mul(v.Len(), v.buffer.ptr, v1.(*vector32).buffer.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1)
}

func (v *vector32) Div(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	v.assertMatch(v1)
	v.ops().Div(v.Len(), v.buffer.ptr, v1.(*vector32).buffer.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1)
}

func (v *vector32) Gemm(transA, transB bool, m, n, k int, alpha anyvec.Numeric, a anyvec.Vector,
	lda int, b anyvec.Vector, ldb int, beta anyvec.Numeric, ldc int) {
	if v == a || v == b {
		panic("output matrix cannot be an input")
	}
	aBuf := a.(*vector32).buffer
	bBuf := b.(*vector32).buffer
	validateGemm(transA, transB, m, n, k, a.Len(), lda, b.Len(), ldb, v.Len(), ldc)
	v.ops().Gemm(transA, transB, m, n, k, alpha.(float32), aBuf.ptr,
		lda, bBuf.ptr, ldb, beta.(float32), v.buffer.ptr, ldc)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(aBuf)
	runtime.KeepAlive(bBuf)
}

func (v *vector32) Exp() {
	v.ops().Exp(v.Len(), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) Tanh() {
	v.ops().Tanh(v.Len(), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) Sin() {
	v.ops().Sin(v.Len(), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) ClipPos() {
	v.ops().ClipPos(v.Len(), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) Sum() anyvec.Numeric {
	return v.aggregate(v.ops().Sum)
}

func (v *vector32) ScaleChunks(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	if v.Len()%v1.Len() != 0 {
		panic("number of scalers must divide vector size")
	}
	chunkSize := v.Len() / v1.Len()
	numChunks := v1.Len()
	v1Buf := v1.(*vector32).buffer
	v.ops().MulChunks(numChunks, chunkSize, v.buffer.ptr, v1Buf.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1Buf)
}

func (v *vector32) AddChunks(v1 anyvec.Vector) {
	if v == v1 {
		panic("arguments cannot be equal")
	}
	if v.Len()%v1.Len() != 0 {
		panic("number of scalers must divide vector size")
	}
	chunkSize := v.Len() / v1.Len()
	numChunks := v1.Len()
	v1Buf := v1.(*vector32).buffer
	v.ops().AddChunks(numChunks, chunkSize, v.buffer.ptr, v1Buf.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1Buf)
}

func (v *vector32) Rand(p anyvec.ProbDist, r *rand.Rand) {
	v.ops().GenRand(v.Len(), v.buffer.ptr, p)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) AddRepeated(v1 anyvec.Vector) {
	if v1.Len() == 0 {
		panic("repeated vector cannot be empty")
	}
	v1Buf := v1.(*vector32).buffer
	v.ops().AddRepeated(v.Len(), v1.Len(), v.buffer.ptr, v1Buf.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1Buf)
}

func (v *vector32) ScaleRepeated(v1 anyvec.Vector) {
	if v1.Len() == 0 {
		panic("repeated vector cannot be empty")
	}
	v1Buf := v1.(*vector32).buffer
	v.ops().ScaleRepeated(v.Len(), v1.Len(), v.buffer.ptr, v1Buf.ptr)
	runtime.KeepAlive(v.buffer)
	runtime.KeepAlive(v1Buf)
}

func (v *vector32) AbsSum() anyvec.Numeric {
	return v.aggregate(v.ops().Asum)
}

func (v *vector32) AbsMax() anyvec.Numeric {
	return v.aggregate(v.ops().Amax)
}

func (v *vector32) Norm() anyvec.Numeric {
	return v.aggregate(v.ops().Nrm2)
}

func (v *vector32) LessThan(n anyvec.Numeric) {
	v.comparison(n, lessThan)
}

func (v *vector32) GreaterThan(n anyvec.Numeric) {
	v.comparison(n, greaterThan)
}

func (v *vector32) EqualTo(n anyvec.Numeric) {
	v.comparison(n, equalTo)
}

func (v *vector32) aggregate(f func(n int, v unsafe.Pointer) float32) anyvec.Numeric {
	res := f(v.Len(), v.buffer.ptr)
	runtime.KeepAlive(v.buffer)
	return res
}

func (v *vector32) comparison(n anyvec.Numeric, c compareType) {
	v.ops().Compare(v.Len(), n.(float32), v.buffer.ptr, c)
	runtime.KeepAlive(v.buffer)
}

func (v *vector32) assertMatch(v1 anyvec.Vector) {
	if v.Len() != v1.Len() {
		panic("sizes do no match")
	}
}

func (v *vector32) ops() ops32 {
	return ops32{h: v.creator.handle}
}

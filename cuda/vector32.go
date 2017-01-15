package cuda

import (
	"math"

	"github.com/unixpickle/anyvec/anyvec32"
)

// A Creator32 implements anyvec32.Creator.
type Creator32 struct {
	handle *Handle
}

// NewCreator32 creates a Creator32 that uses a given CUDA
// handle.
func NewCreator32(h *Handle) *Creator32 {
	return &Creator32{handle: h}
}

// MakeVector creates a zero'd out anyvec32.Vector.
func (c *Creator32) MakeVector(size int) anyvec32.Vector {
	buf, err := newBuffer(c.handle, size*4)
	if err != nil {
		panic(err)
	}
	if err := buf.Clear(); err != nil {
		panic(err)
	}
	return &vector32{
		handle: c.handle,
		buffer: buf,
	}
}

// MakeVectorData creates an anyvec32.Vector with the
// specified contents.
func (c *Creator32) MakeVectorData(d []float32) anyvec32.Vector {
	buf, err := newBuffer(c.handle, len(d)*4)
	if err != nil {
		panic(err)
	}
	if err := buf.Set32(d); err != nil {
		panic(err)
	}
	return &vector32{
		handle: c.handle,
		buffer: buf,
	}
}

type vector32 struct {
	handle *Handle
	buffer *buffer
}

func (v *vector32) Len() int {
	return v.buffer.Len() / 4
}

func (v *vector32) Data() []float32 {
	res := make([]float32, v.Len())
	if err := v.buffer.Get32(res); err != nil {
		panic(err)
	}
	return res
}

func (v *vector32) SetData(d []float32) {
	if err := v.buffer.Set32(d); err != nil {
		panic(err)
	}
}

func (v *vector32) Copy() anyvec32.Vector {
	newBuff, err := newBuffer(v.handle, v.buffer.size)
	if err != nil {
		panic(err)
	}
	if err := newBuff.Set(v.buffer); err != nil {
		panic(err)
	}
	return &vector32{
		handle: v.handle,
		buffer: newBuff,
	}
}

func (v *vector32) Scale(s float32) {
	v.handle.sscal(v.Len(), s, v.buffer.ptr)
}

func (v *vector32) AddScaler(s float32) {
	constVec, err := newBuffer(v.handle, v.buffer.size)
	if err != nil {
		panic(err)
	}
	if err := constVec.SetRepeated32(s); err != nil {
		panic(err)
	}
	v.Add(&vector32{handle: v.handle, buffer: constVec})
}

func (v *vector32) Dot(v1 anyvec32.Vector) float32 {
	v.assertMatch(v1)
	return v.handle.sdot(v.Len(), v.buffer.ptr, v1.(*vector32).buffer.ptr)
}

func (v *vector32) Add(v1 anyvec32.Vector) {
	v.assertMatch(v1)
	v.handle.saxpy(v.Len(), 1, v1.(*vector32).buffer.ptr, v.buffer.ptr)
}

func (v *vector32) Sub(v1 anyvec32.Vector) {
	v.assertMatch(v1)
	v.handle.saxpy(v.Len(), -1, v1.(*vector32).buffer.ptr, v.buffer.ptr)
}

func (v *vector32) Mul(v1 anyvec32.Vector) {
	v.assertMatch(v1)
	v.handle.mul(v.Len(), v.buffer.ptr, v1.(*vector32).buffer.ptr)
}

func (v *vector32) Div(v1 anyvec32.Vector) {
	v.assertMatch(v1)
	// TODO: use custom kernel for this.
	d1 := v.Data()
	d2 := v1.Data()
	for i, x := range d2 {
		d1[i] /= x
	}
	v.SetData(d1)
}

func (v *vector32) Gemm(transA, transB bool, m, n, k int, alpha float32, a anyvec32.Vector, lda int,
	b anyvec32.Vector, ldb int, beta float32, ldc int) {
	validateGemm(transA, transB, m, n, k, a.Len(), lda, b.Len(), ldb, v.Len(), ldc)
	v.handle.sgemm(transA, transB, m, n, k, alpha, a.(*vector32).buffer.ptr,
		lda, b.(*vector32).buffer.ptr, ldb, beta, v.buffer.ptr, ldc)
}

func (v *vector32) Exp() {
	// TODO: use custom kernel for this.
	d := v.Data()
	for i, x := range d {
		d[i] = float32(math.Exp(float64(x)))
	}
	v.SetData(d)
}

func (v *vector32) assertMatch(v1 anyvec32.Vector) {
	if v.Len() != v1.Len() {
		panic("sizes do no match")
	}
}

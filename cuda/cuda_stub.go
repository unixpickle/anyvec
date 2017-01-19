//+build nocuda

package cuda

import (
	"errors"
	"unsafe"

	"github.com/unixpickle/anyvec"
)

type compareType int

const (
	lessThan compareType = iota
	greaterThan
	equalTo
)

// A Handle manages an internal CUDA context.
type Handle struct {
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	return nil, errors.New("unimplemented")
}

type ops32 struct {
	h *Handle
}

func (o ops32) Scal(n int, s float32, x unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Dot(n int, x, y unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (o ops32) Axpy(n int, alpha float32, x, y unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Gemm(transA, transB bool, m, n, k int, alpha float32, a unsafe.Pointer,
	lda int, b unsafe.Pointer, ldb int, beta float32, c unsafe.Pointer, ldc int) {
	panic("unimplemented")
}

func (o ops32) Mul(n int, a, b unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) MulChunks(chunkCount, chunkSize int, vec, scales unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) AddChunks(chunkCount, chunkSize int, vec, scales unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Sum(n int, a unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (o ops32) Div(n int, a, b unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Exp(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Log(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Tanh(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Sin(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) ClipPos(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) GenRand(n int, a unsafe.Pointer, dist anyvec.ProbDist) {
	panic("unimplemented")
}

func (o ops32) AddRepeated(dstLen, srcLen int, dst, src unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) ScaleRepeated(dstLen, srcLen int, dst, src unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) AddScaler(n int, s float32, x unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Asum(n int, v unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (o ops32) Amax(n int, v unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (o ops32) Nrm2(n int, v unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (o ops32) Compare(n int, alpha float32, v unsafe.Pointer, c compareType) {
	panic("unimplemented")
}

func (o ops32) AddLogs(rows, cols int, src unsafe.Pointer) unsafe.Pointer {
	panic("unimplemented")
}

func (o ops32) LogSoftmax(rows, cols int, vecs unsafe.Pointer) {
	panic("unimplemented")
}

func (o ops32) Pow(n int, p float32, vecs unsafe.Pointer) {
	panic("unimplemented")
}

type buffer struct {
	size int
	ptr  unsafe.Pointer
}

func newBufferPtr(h *Handle, size int, buf unsafe.Pointer) *buffer {
	panic("unimplemented")
}

func newBuffer(h *Handle, size int) (*buffer, error) {
	return nil, errors.New("unimplemented")
}

func newBufferConcat(h *Handle, bufs []*buffer) (*buffer, error) {
	return nil, errors.New("unimplemented")
}

func (b *buffer) Len() int {
	panic("unimplemented")
}

func (b *buffer) Clear() error {
	panic("unimplemented")
}

func (b *buffer) Set(b1 *buffer) error {
	panic("unimplemented")
}

func (b *buffer) Set32(source []float32) error {
	panic("unimplemented")
}

func (b *buffer) Set64(source []float64) error {
	panic("unimplemented")
}

func (b *buffer) Get32(source []float32) error {
	panic("unimplemented")
}

func (b *buffer) Get64(source []float64) error {
	panic("unimplemented")
}

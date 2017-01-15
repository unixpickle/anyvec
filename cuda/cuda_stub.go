//+build nocuda

package cuda

import (
	"errors"
	"unsafe"
)

// These errors indicate various CUDA-related failures.
var (
	ErrHandleCreation = errors.New("failed to create cuBLAS handle")
	ErrMemoryAlloc    = errors.New("failed to allocate CUDA memory")
	ErrMemoryZero     = errors.New("failed to zero CUDA memory")
	ErrMemorySet      = errors.New("failed to set CUDA memory")
	ErrMemoryCopy     = errors.New("failed to transfer CUDA memory")
	ErrMatrixMultiply = errors.New("matrix multiplication failed")
)

// A Handle manages an internal CUDA context.
type Handle struct {
}

// NewHandle attempts to get a new Handle.
func NewHandle() (*Handle, error) {
	return nil, errors.New("unimplemented")
}

func (h *Handle) sscal(n int, s float32, x unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) sdot(n int, x, y unsafe.Pointer) float32 {
	panic("unimplemented")
}

func (h *Handle) saxpy(n int, alpha float32, x, y unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) sgemm(transA, transB bool, m, n, k int, alpha float32, a unsafe.Pointer,
	lda int, b unsafe.Pointer, ldb int, beta float32, c unsafe.Pointer, ldc int) {
	panic("unimplemented")
}

func (h *Handle) mul(n int, a, b unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) div(n int, a, b unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) exp(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) tanh(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) sin(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

func (h *Handle) clipPos(n int, a unsafe.Pointer) {
	panic("unimplemented")
}

// A buffer is an on-device memory buffer.
type buffer struct {
	size int
	ptr  unsafe.Pointer
}

// newBuffer allocates a buffer.
func newBuffer(h *Handle, size int) (*buffer, error) {
	return nil, errors.New("unimplemented")
}

// Len returns the buffer's length in bytes.
func (b *buffer) Len() int {
	panic("unimplemented")
}

// Clear zeroes the buffer.
func (b *buffer) Clear() error {
	panic("unimplemented")
}

// Set copies the contents of a buffer into b.
func (b *buffer) Set(b1 *buffer) error {
	panic("unimplemented")
}

// Set32 copies 32-bit floats into the buffer.
func (b *buffer) Set32(source []float32) error {
	panic("unimplemented")
}

// SetRepeated32 copies the same 32-bits again and again
// to fill the buffer.
func (b *buffer) SetRepeated32(v float32) error {
	panic("unimplemented")
}

// Set64 copies 64-bit floats into the buffer.
func (b *buffer) Set64(source []float64) error {
	panic("unimplemented")
}

// SetRepeated64 copies the same 64-bits again and again
// to fill the buffer.
func (b *buffer) SetRepeated64(v float64) error {
	panic("unimplemented")
}

// Get32 copies 32-bit floats out of the buffer.
func (b *buffer) Get32(source []float32) error {
	panic("unimplemented")
}

// Get64 copies 64-bit floats out of the buffer.
func (b *buffer) Get64(source []float64) error {
	panic("unimplemented")
}

//+build nocuda

package cuda

import (
	"errors"

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

// SetPoolSize tells the Handle how much CUDA memory it
// should cache, in bytes.
//
// The buffer pool is useful for applications that tend to
// allocate vectors of the same length over and over.
// For applications that allocate many different vector
// sizes, the buffer pool will not help very much.
//
// The default pool size is 0.
func (h *Handle) SetPoolSize(size int64) {
	panic("unavailable with nocuda tag")
}

// A Creator32 implements anyvec.Creator for float32
// numerics.
type Creator32 struct {
	handle *Handle
}

// NewCreator32 creates a Creator32 that uses a given CUDA
// handle.
func NewCreator32(h *Handle) *Creator32 {
	panic("unavailable with nocuda tag")
}

// MakeNumeric creates a float32.
func (c *Creator32) MakeNumeric(x float64) anyvec.Numeric {
	panic("unavailable with nocuda tag")
}

// MakeNumericList creates a []float32.
func (c *Creator32) MakeNumericList(x []float64) anyvec.NumericList {
	panic("unavailable with nocuda tag")
}

// MakeVector creates a zero'd out anyvec.Vector.
func (c *Creator32) MakeVector(size int) anyvec.Vector {
	panic("unavailable with nocuda tag")
}

// MakeVectorData creates an anyvec.Vector with the
// specified contents.
func (c *Creator32) MakeVectorData(dObj anyvec.NumericList) anyvec.Vector {
	panic("unavailable with nocuda tag")
}

// Concat concatenates vectors.
func (c *Creator32) Concat(v ...anyvec.Vector) anyvec.Vector {
	panic("unavailable with nocuda tag")
}

// Package anyvec64 provides an anyvec implementation that
// uses float64 numerics.
package anyvec64

import (
	"sync"

	"github.com/unixpickle/anyvec"
)

var curCreator anyvec.Creator = DefaultCreator{}
var curCreatorLock sync.RWMutex

// MakeNumeric creates a numeric using the current
// creator.
func MakeNumeric(x float64) anyvec.Numeric {
	return CurrentCreator().MakeNumeric(x)
}

// MakeNumericList creates a list of numerics using the
// current creator.
func MakeNumericList(x []float64) []float64 {
	return CurrentCreator().MakeNumericList(x).([]float64)
}

// MakeVector creates a vector using the current creator.
// The Vector is initially set to all zeros.
func MakeVector(size int) anyvec.Vector {
	return CurrentCreator().MakeVector(size)
}

// MakeVectorData creates a vector using the current
// creator.
// The Vector is initialized to the passed data.
func MakeVectorData(data []float64) anyvec.Vector {
	return CurrentCreator().MakeVectorData(data)
}

// Concat concatenates zero or more vectors using the
// current creator.
// The resulting vector should contain a complete copy of
// the original vectors, so that the original vectors can
// be modified without changing the new vector.
func Concat(vs ...anyvec.Vector) anyvec.Vector {
	return CurrentCreator().Concat(vs...)
}

// MakeMapper creates a Mapper using the current Creator.
func MakeMapper(inSize int, table []int) anyvec.Mapper {
	return CurrentCreator().MakeMapper(inSize, table)
}

// Use sets the current anyvec.Creator.
// The creator must use float64 numerics.
func Use(c anyvec.Creator) {
	if _, ok := c.MakeNumeric(3.14).(float64); !ok {
		panic("creator does not use float64 numerics")
	}
	curCreatorLock.Lock()
	curCreator = c
	curCreatorLock.Unlock()
}

// CurrentCreator gets the current Creator.
func CurrentCreator() anyvec.Creator {
	curCreatorLock.RLock()
	c := curCreator
	curCreatorLock.RUnlock()
	return c
}

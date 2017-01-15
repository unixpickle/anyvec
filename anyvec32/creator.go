package anyvec32

import "sync"

var curCreator Creator = DefaultCreator{}
var curCreatorLock sync.RWMutex

// A Creator creates Vectors.
//
// See the corresponding package-level functions for
// details on the individual methods.
type Creator interface {
	MakeVector(size int) Vector
	MakeVectorData(data []float32) Vector
	Concat(vs ...Vector) Vector
}

// MakeVector creates a vector using the current creator.
// The Vector is initially set to all zeros.
func MakeVector(size int) Vector {
	return CurrentCreator().MakeVector(size)
}

// MakeVectorData creates a vector using the current
// creator.
// The Vector is initialized to the passed data.
func MakeVectorData(data []float32) Vector {
	return CurrentCreator().MakeVectorData(data)
}

// Concat concatenates zero or more vectors using the
// current creator.
// The resulting vector should contain a complete copy of
// the original vectors, so that the original vectors can
// be modified without changing the new vector.
func Concat(vs ...Vector) Vector {
	return CurrentCreator().Concat(vs...)
}

// Use sets the current Creator.
func Use(c Creator) {
	curCreatorLock.Lock()
	curCreator = c
	curCreatorLock.Unlock()
}

// CurrentCreator gets the current Creator.
func CurrentCreator() Creator {
	curCreatorLock.RLock()
	c := curCreator
	curCreatorLock.RUnlock()
	return c
}

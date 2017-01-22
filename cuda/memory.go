//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
*/
import "C"

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"unsafe"
)

const minGCThresh = 1 << 15

// gc manages garbage collection.
type gc struct {
	usage        int
	thresh       int
	increaseFrac float64
}

func newGC() *gc {
	frac := 2.0
	if per := os.Getenv("GOGC"); per != "" {
		if perVal, err := strconv.Atoi(per); err != nil {
			frac = 1 + float64(perVal)/100.0
		}
	}
	return &gc{
		thresh:       minGCThresh,
		increaseFrac: frac,
	}
}

func (g *gc) Alloc(amount int) {
	g.usage += amount
	if g.usage > g.thresh {
		g.thresh = int(float64(g.usage) * g.increaseFrac)
		runtime.GC()
	}
}

func (g *gc) Free(amount int) {
	g.usage -= amount
	if g.usage < 0 {
		panic(fmt.Sprintf("invalid memory used: %d", g.usage))
	}
	newThresh := int(float64(g.usage) * g.increaseFrac)
	if newThresh < g.thresh {
		g.thresh = newThresh
		if g.thresh < minGCThresh {
			g.thresh = minGCThresh
		}
	}
}

// bufferPool is a quick and dirty memory buffer cache to
// avoid expensive cudaMalloc() calls.
//
// All methods of bufferPool should be called from the
// CUDA OS thread.
type bufferPool struct {
	capacity int64
	live     int64
	cached   int64

	cache map[int][]unsafe.Pointer
}

func newBufferPool() *bufferPool {
	return &bufferPool{cache: map[int][]unsafe.Pointer{}}
}

func (b *bufferPool) SetCapacity(c int64) {
	b.capacity = c
	b.evictAsNeeded()
}

func (b *bufferPool) Alloc(size int) (unsafe.Pointer, error) {
	if bufs, ok := b.cache[size]; ok {
		buf := bufs[len(bufs)-1]
		if len(bufs) == 1 {
			delete(b.cache, size)
		} else {
			b.cache[size] = bufs[:len(bufs)-1]
		}
		b.cached -= int64(size)
		b.live += int64(size)
		if b.cached < 0 {
			panic("internal cache inconsistency")
		}
		return buf, nil
	}

	b.live += int64(size)
	b.evictAsNeeded()

	var newBuf unsafe.Pointer
	err := cudaError("cudaMalloc", C.cudaMalloc(&newBuf, C.size_t(size)))
	return newBuf, err
}

func (b *bufferPool) Free(buf unsafe.Pointer, size int) {
	if b.capacity == 0 {
		b.live -= int64(size)
		C.cudaFree(buf)
		return
	}
	b.cache[size] = append(b.cache[size], buf)
	b.cached += int64(size)
	b.live -= int64(size)
	if b.live < 0 {
		panic("internal cache inconsistency")
	}
	b.evictAsNeeded()
}

func (b *bufferPool) Destroy() {
	for _, list := range b.cache {
		for _, ptr := range list {
			C.cudaFree(ptr)
		}
	}
}

func (b *bufferPool) evictAsNeeded() {
	for b.cached > 0 && b.cached+b.live > b.capacity {
		// We choose a random bin to prune rather than using an
		// algorithm like "least recently used".
		// The random approach has some merits, like avoiding
		// pathological round-robin cases.

		var sizes []int
		for size := range b.cache {
			if size != 0 {
				sizes = append(sizes, size)
			}
		}

		pruneSize := sizes[rand.Intn(len(sizes))]
		bin := b.cache[pruneSize]
		last := bin[len(bin)-1]
		C.cudaFree(last)
		b.cached -= int64(pruneSize)
		if len(bin) == 1 {
			delete(b.cache, pruneSize)
		} else {
			b.cache[pruneSize] = bin[:len(bin)-1]
		}
	}
}

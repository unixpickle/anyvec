package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
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
		if perVal, err := strconv.Atoi(per); err == nil {
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

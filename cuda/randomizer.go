//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "curand.h"

const curandRngType_t generatorType = CURAND_RNG_PSEUDO_DEFAULT;
const curandStatus_t curandSuccess = CURAND_STATUS_SUCCESS;
const curandGenerator_t nullGen = NULL;
*/
import "C"
import (
	"errors"
	"fmt"
	"time"
	"unsafe"
)

// A randomizer samples random numbers.
type randomizer struct {
	gen C.curandGenerator_t
}

// newRandomizer creates a new randomizer.
func newRandomizer() (*randomizer, error) {
	var gen C.curandGenerator_t
	res := C.curandCreateGenerator(&gen, C.generatorType)
	if res != C.curandSuccess {
		return nil, errors.New("initialize cuRAND failed")
	}
	seed := C.ulonglong(time.Now().UnixNano())
	res = C.curandSetPseudoRandomGeneratorSeed(gen, seed)
	if res != C.curandSuccess {
		C.curandDestroyGenerator(gen)
		return nil, errors.New("seed cuRAND failed")
	}
	return &randomizer{gen: gen}, nil
}

// Uniform32 creates uniform random values.
func (r *randomizer) Uniform32(k *mathKernels, dest unsafe.Pointer, n int) error {
	res := C.curandGenerateUniform(r.gen, (*C.float)(dest), C.size_t(n))
	if res != C.curandSuccess {
		return fmt.Errorf("cuRAND uniform sampling failed: %d", int(res))
	}
	return k.ShiftRandUniform32(dest, n)
}

// Norm32 creates normally distributed random values.
func (r *randomizer) Norm32(dest unsafe.Pointer, n int) error {
	var res C.curandStatus_t
	// cuRAND requires the size to be a multiple of 2.
	if n%2 == 0 {
		res = C.curandGenerateNormal(r.gen, (*C.float)(dest), C.size_t(n), 0, 1)
	} else {
		var temp unsafe.Pointer
		if C.cudaMalloc(&temp, C.size_t((n+1)*4)) != C.cudaSuccess {
			return ErrMemoryAlloc
		}
		defer C.cudaFree(temp)
		res = C.curandGenerateNormal(r.gen, (*C.float)(temp), C.size_t(n+1), 0, 1)
		if res == C.curandSuccess {
			cpyRes := C.cudaMemcpy(dest, temp, C.size_t(n*4), C.cudaMemcpyDeviceToDevice)
			if cpyRes != C.cudaSuccess {
				return ErrMemoryCopy
			}
		}
	}
	if res != C.curandSuccess {
		return fmt.Errorf("cuRAND normal sampling failed: %d", int(res))
	}
	return nil
}

// Bernoulli32 generates bernoulli random variables.
func (r *randomizer) Bernoulli32(k *mathKernels, dest unsafe.Pointer, n int) error {
	res := C.curandGenerateUniform(r.gen, (*C.float)(dest), C.size_t(n))
	if res != C.curandSuccess {
		return fmt.Errorf("cuRAND uniform sampling failed: %d", int(res))
	}
	return k.UniformToBernoulli32(dest, n)
}

// Destroy releases the randomizer's resources.
func (r *randomizer) Destroy() {
	C.curandDestroyGenerator(r.gen)
}

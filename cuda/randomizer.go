//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "curand.h"

const curandRngType_t generatorType = CURAND_RNG_PSEUDO_DEFAULT;
*/
import "C"
import (
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
	if err := curandError("curandCreateGenerator", res); err != nil {
		return nil, err
	}
	seed := C.ulonglong(time.Now().UnixNano())
	res = C.curandSetPseudoRandomGeneratorSeed(gen, seed)
	if err := curandError("curandSetPseudoRandomGeneratorSeed", res); err != nil {
		C.curandDestroyGenerator(gen)
		return nil, err
	}
	return &randomizer{gen: gen}, nil
}

// Uniform32 creates uniform random values.
func (r *randomizer) Uniform32(k *mathKernels, dest unsafe.Pointer, n int) error {
	res := C.curandGenerateUniform(r.gen, (*C.float)(dest), C.size_t(n))
	if err := curandError("curandGenerateUniform", res); err != nil {
		return err
	}
	return k.ShiftRandUniform32(dest, n)
}

// Norm32 creates normally distributed random values.
func (r *randomizer) Norm32(dest unsafe.Pointer, n int) error {
	// cuRAND requires the size to be a multiple of 2.
	if n%2 == 0 {
		res := C.curandGenerateNormal(r.gen, (*C.float)(dest), C.size_t(n), 0, 1)
		return curandError("curandGenerateNormal", res)
	}

	var temp unsafe.Pointer
	err := cudaError("cudaMalloc", C.cudaMalloc(&temp, C.size_t((n+1)*4)))
	if err != nil {
		return err
	}
	defer C.cudaFree(temp)

	res := C.curandGenerateNormal(r.gen, (*C.float)(temp), C.size_t(n+1), 0, 1)
	if err := curandError("curandGenerateNormal", res); err != nil {
		return err
	}

	cpyRes := C.cudaMemcpy(dest, temp, C.size_t(n*4), C.cudaMemcpyDeviceToDevice)
	return cudaError("cudaMemcpy", cpyRes)
}

// Bernoulli32 generates bernoulli random variables.
func (r *randomizer) Bernoulli32(k *mathKernels, dest unsafe.Pointer, n int) error {
	res := C.curandGenerateUniform(r.gen, (*C.float)(dest), C.size_t(n))
	if err := curandError("curandGenerateUniform", res); err != nil {
		return err
	}
	return k.UniformToBernoulli32(dest, n)
}

// Destroy releases the randomizer's resources.
func (r *randomizer) Destroy() {
	C.curandDestroyGenerator(r.gen)
}

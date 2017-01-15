//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvrtc.h"

extern CUresult cuSuccess;
const nvrtcResult nvrtcSuccess = NVRTC_SUCCESS;
const char ** nullStrPtr = NULL;
const CUjit_option * nullJitOptions = NULL;
const void ** nullPtrPtr = NULL;

CUresult anyvec_cuda_callkernel(CUfunction f, void * p1, void * p2, size_t n) {
	void * args[] = {&p1, &p2, &n};
	// TODO: look into these constants.
	return cuLaunchKernel(f, 32, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

type mathKernels struct {
	module  C.CUmodule
	kernels map[string]C.CUfunction
	prog    C.nvrtcProgram
}

func newMathKernels() (kernels *mathKernels, err error) {
	code := C.CString(mathKernelsCode)
	defer C.free(unsafe.Pointer(code))
	fileName := C.CString("prog.cu")
	defer C.free(unsafe.Pointer(fileName))

	var prog C.nvrtcProgram
	res := C.nvrtcCreateProgram(&prog, code, fileName, 0,
		C.nullStrPtr, C.nullStrPtr)
	if err := nvrtcError("nvrtcCreateProgram", res); err != nil {
		return nil, err
	}

	defer func() {
		if kernels == nil {
			C.nvrtcDestroyProgram(&prog)
		}
	}()

	for _, name := range mathKernelNames {
		cName := C.CString(name)
		res = C.nvrtcAddNameExpression(prog, cName)
		C.free(unsafe.Pointer(cName))
		if err := nvrtcError("nvrtcAddNameExpression", res); err != nil {
			return nil, err
		}
	}
	res = C.nvrtcCompileProgram(prog, 0, C.nullStrPtr)
	if err := nvrtcError("nvrtcCompileProgram", res); err != nil {
		return nil, err
	}
	var ptxSize C.size_t
	res = C.nvrtcGetPTXSize(prog, &ptxSize)
	if err := nvrtcError("nvrtcGetPTXSize", res); err != nil {
		return nil, err
	}
	ptx := C.malloc(ptxSize)
	defer C.free(ptx)
	res = C.nvrtcGetPTX(prog, (*C.char)(ptx))
	if err := nvrtcError("nvrtcGetPTX", res); err != nil {
		return nil, err
	}

	var module C.CUmodule
	cuRes := C.cuModuleLoadDataEx(&module, ptx, 0, C.nullJitOptions, C.nullPtrPtr)
	if cuRes != C.cuSuccess {
		return nil, errors.New("cuModuleLoadDataEx failed")
	}
	defer func() {
		if kernels == nil {
			C.cuModuleUnload(module)
		}
	}()

	kernelMap := map[string]C.CUfunction{}
	for _, nameStr := range mathKernelNames {
		var name *C.char
		origName := C.CString(nameStr)
		defer C.free(unsafe.Pointer(origName))
		res = C.nvrtcGetLoweredName(prog, origName, &name)
		if err := nvrtcError("nvrtcGetLoweredName", res); err != nil {
			return nil, err
		}
		var kernel C.CUfunction
		cuRes := C.cuModuleGetFunction(&kernel, module, name)
		if cuRes != C.cuSuccess {
			return nil, errors.New("cuModuleGetFunction failed")
		}
		kernelMap[nameStr] = kernel
	}

	return &mathKernels{
		module:  module,
		kernels: kernelMap,
		prog:    prog,
	}, nil
}

func (m *mathKernels) Destroy() {
	C.cuModuleUnload(m.module)
	C.nvrtcDestroyProgram(&m.prog)
}

// Div32 performs element-wise division.
func (m *mathKernels) Div32(num, denom unsafe.Pointer, n int) error {
	k := m.kernels["divElements"]
	res := C.anyvec_cuda_callkernel(k, num, denom, C.size_t(n))
	if res != C.cuSuccess {
		return errors.New("cuLaunchKernel failed")
	}
	res = C.cuCtxSynchronize()
	if res != C.cuSuccess {
		return errors.New("cuCtxSynchronize failed")
	}
	return nil
}

func nvrtcError(funcName string, status C.nvrtcResult) error {
	if status != C.nvrtcSuccess {
		return errors.New(funcName + " failed")
	}
	return nil
}

var mathKernelNames = []string{"divElements"}

const mathKernelsCode string = `
__global__ void divElements(float * x, float * y, size_t n) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		x[tid] /= y[tid];
	}
}
`

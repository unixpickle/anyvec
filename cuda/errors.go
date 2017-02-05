package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "curand.h"

const char * nullMessage = NULL;

const char * anyvec_cu_err(CUresult res) {
	switch (res) {
	case CUDA_SUCCESS:
		return NULL;
	case CUDA_ERROR_INVALID_VALUE:
		return "CUDA_ERROR_INVALID_VALUE";
	case CUDA_ERROR_OUT_OF_MEMORY:
		return "CUDA_ERROR_OUT_OF_MEMORY";
	case CUDA_ERROR_NOT_INITIALIZED:
		return "CUDA_ERROR_NOT_INITIALIZED";
	case CUDA_ERROR_DEINITIALIZED:
		return "CUDA_ERROR_DEINITIALIZED";
	case CUDA_ERROR_NO_DEVICE:
		return "CUDA_ERROR_NO_DEVICE";
	case CUDA_ERROR_INVALID_DEVICE:
		return "CUDA_ERROR_INVALID_DEVICE";
	case CUDA_ERROR_INVALID_IMAGE:
		return "CUDA_ERROR_INVALID_IMAGE";
	case CUDA_ERROR_INVALID_CONTEXT:
		return "CUDA_ERROR_INVALID_CONTEXT";
	case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
		return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
	case CUDA_ERROR_MAP_FAILED:
		return "CUDA_ERROR_MAP_FAILED";
	case CUDA_ERROR_UNMAP_FAILED:
		return "CUDA_ERROR_UNMAP_FAILED";
	case CUDA_ERROR_ARRAY_IS_MAPPED:
		return "CUDA_ERROR_ARRAY_IS_MAPPED";
	case CUDA_ERROR_ALREADY_MAPPED:
		return "CUDA_ERROR_ALREADY_MAPPED";
	case CUDA_ERROR_NO_BINARY_FOR_GPU:
		return "CUDA_ERROR_NO_BINARY_FOR_GPU";
	case CUDA_ERROR_ALREADY_ACQUIRED:
		return "CUDA_ERROR_ALREADY_ACQUIRED";
	case CUDA_ERROR_NOT_MAPPED:
		return "CUDA_ERROR_NOT_MAPPED";
	case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
		return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
	case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
		return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
	case CUDA_ERROR_ECC_UNCORRECTABLE:
		return "CUDA_ERROR_ECC_UNCORRECTABLE";
	case CUDA_ERROR_UNSUPPORTED_LIMIT:
		return "CUDA_ERROR_UNSUPPORTED_LIMIT";
	case CUDA_ERROR_INVALID_SOURCE:
		return "CUDA_ERROR_INVALID_SOURCE";
	case CUDA_ERROR_FILE_NOT_FOUND:
		return "CUDA_ERROR_FILE_NOT_FOUND";
	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
		return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
		return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
	case CUDA_ERROR_OPERATING_SYSTEM:
		return "CUDA_ERROR_OPERATING_SYSTEM";
	case CUDA_ERROR_INVALID_HANDLE:
		return "CUDA_ERROR_INVALID_HANDLE";
	case CUDA_ERROR_NOT_FOUND:
		return "CUDA_ERROR_NOT_FOUND";
	case CUDA_ERROR_NOT_READY:
		return "CUDA_ERROR_NOT_READY";
	case CUDA_ERROR_LAUNCH_FAILED:
		return "CUDA_ERROR_LAUNCH_FAILED";
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
		return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
	case CUDA_ERROR_LAUNCH_TIMEOUT:
		return "CUDA_ERROR_LAUNCH_TIMEOUT";
	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
		return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
	default:
		return "CUDA_ERROR_UNKNOWN";
	}
}

const char * anyvec_cublas_err(cublasStatus_t s) {
	switch (s) {
	case CUBLAS_STATUS_SUCCESS:
		return NULL;
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	default:
		return "unknown cuBLAS error";
	}
}

const char * anyvec_curand_err(curandStatus_t s) {
	switch (s) {
	case CURAND_STATUS_SUCCESS:
		return NULL;
	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	default:
		return "unknown cuRAND error";
	}
}
*/
import "C"
import "errors"

// cuError returns an error for a CUresult, or nil
// if there is no error.
func cuError(context string, e C.CUresult) error {
	return errForCStr(context, C.anyvec_cu_err(e))
}

// cudaError returns an error for a cudaError_t, or nil if
// there is no error.
func cudaError(context string, e C.cudaError_t) error {
	if e == C.cudaSuccess {
		return nil
	}
	return errors.New(context + ": " + C.GoString(C.cudaGetErrorString(e)))
}

// cublasError returns an error for a cublasStatus_t, or
// nil if there is no error.
func cublasError(context string, e C.cublasStatus_t) error {
	return errForCStr(context, C.anyvec_cublas_err(e))
}

// curandError returns an error for a curandStatus_t, or
// nil if there is no error.
func curandError(context string, e C.curandStatus_t) error {
	return errForCStr(context, C.anyvec_curand_err(e))
}

// must panics if the error is non-nil.
func must(e error) {
	if e != nil {
		panic(e)
	}
}

// mustBLAS panics with an appropriate error message if a
// BLAS API fails.
func mustBLAS(context string, s C.cublasStatus_t) {
	must(cublasError(context, s))
}

func errForCStr(context string, msg *C.char) error {
	if msg != C.nullMessage {
		return errors.New(context + ": " + C.GoString(msg))
	} else {
		return nil
	}
}

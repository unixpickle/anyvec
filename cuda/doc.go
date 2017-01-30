// Package cuda provides a hardware-accelerated anyvec
// implementation.
//
// To use this package, you must tell Go how to link with
// CUDA.
// On Mac OS X, this might look like:
//
//     $ export CUDA_PATH="/Developer/NVIDIA/CUDA-8.0"
//     $ export DYLD_LIBRARY_PATH="$CUDA_PATH/lib":$DYLD_LIBRARY_PATH
//     $ export CPATH="$CUDA_PATH/include/"
//     $ export CGO_LDFLAGS="/usr/local/cuda/lib/libcuda.dylib $CUDA_PATH/lib/libcudart.dylib $CUDA_PATH/lib/libcublas.dylib $CUDA_PATH/lib/libcurand.dylib"
//
// On Linux, this might look like:
//
//     $ export CUDA_PATH=/usr/local/cuda
//     $ export CPATH="$CUDA_PATH/include/"
//     $ export CGO_LDFLAGS="$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcurand.so"
//     $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64/
//
// Once you have your environment configured to build with
// CUDA, you can set anyvec32 to use CUDA like so:
//
//     import (
//         "github.com/unixpickle/anyvec/anyvec32"
//         "github.com/unixpickle/anyvec/cuda"
//     )
//
//     func init() {
//         handle, err := cuda.NewHandle()
//         if err != nil {
//             panic(err)
//         }
//         anyvec32.Use(cuda.NewCreator32(handle))
//     }
//
package cuda

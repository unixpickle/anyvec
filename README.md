# anyvec

**anyvec** lets you use any implementation of vector arithmetic that you want. You might use it for GPU acceleration, distributed matrix multiplication, or any number of things that nobody's thought of yet.

# Installation

Download the code as follows:

```
$ go get -u -tags nocuda github.com/unixpickle/anyvec/...
```

To use the CUDA package, you have to tell the compiler about your CUDA installation. On OS X, this might look like:

```
$ export CUDA_PATH="/Developer/NVIDIA/CUDA-8.0"
$ export DYLD_LIBRARY_PATH="$CUDA_PATH/lib":$DYLD_LIBRARY_PATH
$ export CPATH="$CUDA_PATH/include/"
$ export CGO_LDFLAGS="/usr/local/cuda/lib/libcuda.dylib $CUDA_PATH/lib/libcudart.dylib $CUDA_PATH/lib/libcublas.dylib $CUDA_PATH/lib/libnvrtc.dylib $CUDA_PATH/lib/libcurand.dylib"
```

On Linux, the environment setup might be more like this:

```
$ export CUDA_PATH=/usr/local/cuda
$ export CPATH="$CUDA_PATH/include/"
$ export CGO_LDFLAGS="$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libnvrtc.so $CUDA_PATH/lib64/libcurand.so"
$ export LD_LIBRARY_PATH=$CUDA_PATH/lib64/
```

Once you have this setup, you can enable the CUDA library in `anyvec32` as follows:

```go
import (
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/cuda"
)

func main() {
	handle, err := cuda.NewHandle()
	if err != nil {
		panic(err)
	}
	anyvec32.Use(cuda.NewCreator32(handle))
}
```

# Why not BLAS?

Why would you need anyvec when the tried and true BLAS APIs provide functions for matrix multiplication, vector addition, and the like? One reason is that BLAS assumes too much about where your vectors are stored. BLAS routines take memory addresses as arguments, meaning that your vectors need to be stored in main memory. Some frameworks, like CUDA 8, deal with this using fancy virtual memory mechanisms. However, that is not always an option, especially in higher-level languages like Go.

Unlike BLAS, anyvec uses abstract vector types that don't depend on memory addresses. This makes it possible to offload your vector computations to *anything*, whether it be the GPU or a remote server.

# TODO

 * Amax and Asum (for inf-norm and 1-norm)
 * Euclidean norm
 * AddChunks
 * ScaleRepeated
 * AddLogDomain
 * On-GPU implementation of Max()
 * Kernelized AddScaler()
 * Kernelized Sum()
 * Kernelized LogSoftmax()

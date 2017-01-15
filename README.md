# anyvec

**anyvec** lets you use any implementation of vector arithmetic that you want. You might use it for GPU acceleration, distributed matrix multiplication, or any number of things that nobody's thought of yet.

# Usage

Download the code as follows:

```
$ go get -u -tags nocuda github.com/unixpickle/anyvec/...
```

See [gocublas](https://github.com/unixpickle/gocublas#usage) for details on using the CUDA sub-package.

# Why not BLAS?

Why would you need anyvec when the tried and true BLAS APIs provide functions for matrix multiplication, vector addition, and the like? One reason is that BLAS assumes too much about where your vectors are stored. BLAS routines take memory addresses as arguments, meaning that your vectors need to be stored in main memory. Some frameworks, like CUDA 8, deal with this using fancy virtual memory mechanisms. However, that is not always an option, especially in higher-level languages like Go.

Unlike BLAS, anyvec uses abstract vector types that don't depend on memory addresses. This makes it possible to offload your vector computations to *anything*, whether it be the GPU or a remote server.

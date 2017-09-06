# anyvec [![GoDoc](https://godoc.org/github.com/unixpickle/anyvec?status.svg)](https://godoc.org/github.com/unixpickle/anyvec)

**anyvec** lets you use any implementation of vector arithmetic that you want. You might use it for GPU acceleration, distributed matrix multiplication, or precision-agnostic arithmetic.

**anyvec** should help you do the following:

 * Benefit from GPU hardware acceleration
 * Write code once that works with any floating-point precision
 * Perform common linear algebra routines

# Downloading

First, you must have [Go](https://golang.org/doc/install) installed and configured. Once you do, you can download **anyvec** and its dependencies as follows:

```
$ go get -u -d github.com/unixpickle/anyvec/...
```

# Using CUDA

If you plan on using CUDA, see [cudavec](https://github.com/unixpickle/cudavec).

# Why not BLAS?

Why would you need anyvec when the tried and true BLAS APIs provide functions for matrix multiplication, vector addition, and the like? One reason is that BLAS assumes too much about where your vectors are stored. BLAS routines take memory addresses as arguments, meaning that your vectors need to be stored in main memory. Some frameworks, like CUDA 8, deal with this using fancy virtual memory mechanisms. However, that is not always an option, especially in higher-level languages like Go.

Unlike BLAS, anyvec uses abstract vector types that don't depend on memory addresses. This makes it possible to offload your vector computations to *anything*, whether it be the GPU or a remote server.

# TODO

Here are some things I'd like to add to anyvec at some point:

 * Clip between arbitrary vectors

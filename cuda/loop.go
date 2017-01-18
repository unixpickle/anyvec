//+build !nocuda

package cuda

/*
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
*/
import "C"
import (
	"runtime"
	"sync"
)

var mainLoop *cudaLoop
var mainLoopLock sync.RWMutex

func createMainLoop() error {
	mainLoopLock.Lock()
	defer mainLoopLock.Unlock()
	if mainLoop != nil {
		return nil
	}
	var err error
	mainLoop, err = newCudaLoop()
	return err
}

func getMainLoop() *cudaLoop {
	mainLoopLock.RLock()
	res := mainLoop
	mainLoopLock.RUnlock()
	return res
}

// A cudaLoop runs functions in a dedicated CUDA thread.
type cudaLoop struct {
	ch chan<- *cudaLoopMsg
}

// newCudaLoop creates a new cudaLoop and all of the
// resources associated with it.
func newCudaLoop() (*cudaLoop, error) {
	ch := make(chan *cudaLoopMsg, 1)
	res := &cudaLoop{ch: ch}
	resChan := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		s, err := newCudaState()
		resChan <- err
		if err != nil {
			return
		}
		defer s.destroy()
		cudaLoopMain(ch, s)
	}()
	if err := <-resChan; err != nil {
		return nil, err
	}
	return res, nil
}

// Close ends the loop and destroys its resources.
// This should only be called once.
func (c *cudaLoop) Close() {
	close(c.ch)
}

// Run evaluates the function on the CUDA thread and waits
// for the function to complete.
func (c *cudaLoop) Run(f func()) {
	res := make(chan struct{}, 1)
	c.ch <- &cudaLoopMsg{
		doneChan: res,
		f:        f,
	}
	<-res
}

// RunCUBLAS is like Run, but the function is given access
// to a cuBLAS handle.
func (c *cudaLoop) RunCUBLAS(f func(h C.cublasHandle_t)) {
	res := make(chan struct{}, 1)
	c.ch <- &cudaLoopMsg{
		doneChan: res,
		cublasF:  f,
	}
	<-res
}

type cudaLoopMsg struct {
	doneChan chan<- struct{}
	f        func()
	cublasF  func(h C.cublasHandle_t)
}

type cudaState struct {
	blas C.cublasHandle_t
	ctx  C.CUcontext
}

func newCudaState() (*cudaState, error) {
	C.cuInit(0)

	var dev C.CUdevice
	err := cuError("cuDeviceGet", C.cuDeviceGet((*C.CUdevice)(&dev), 0))
	if err != nil {
		return nil, err
	}

	var ctx C.CUcontext
	err = cuError("cuCtxCreate", C.cuCtxCreate((*C.CUcontext)(&ctx), C.uint(0), dev))
	if err != nil {
		return nil, err
	}

	var handle C.cublasHandle_t
	err = cublasError("cublasCreate", C.cublasCreate(&handle))
	if err != nil {
		C.cuCtxDestroy(ctx)
		return nil, err
	}

	return &cudaState{
		blas: handle,
		ctx:  ctx,
	}, nil
}

func (c *cudaState) destroy() {
	C.cublasDestroy(c.blas)
	C.cuCtxDestroy(c.ctx)
}

func cudaLoopMain(ch <-chan *cudaLoopMsg, s *cudaState) {
	for msg := range ch {
		if msg.f != nil {
			msg.f()
		} else {
			msg.cublasF(s.blas)
		}
		close(msg.doneChan)
	}
}

.PHONY: clean

kernel_code.go: kernels.cu
	nvcc --gpu-architecture=compute_30 --gpu-code=compute_30 --ptx kernels.cu
	echo 'package cuda' >kernel_code.go
	echo '' >>kernel_code.go
	echo 'var kernelPTX = `' >>kernel_code.go
	cat kernels.ptx >>kernel_code.go
	echo '`' >>kernel_code.go
	rm kernels.ptx

clean:
	rm kernel_code.go

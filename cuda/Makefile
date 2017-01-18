.PHONY: clean

kernel_code.go:
	nvcc --ptx kernels.cu
	echo '//+build !nocuda' >kernel_code.go
	echo '' >>kernel_code.go
	echo 'package cuda' >>kernel_code.go
	echo '' >>kernel_code.go
	echo 'var kernelPTX = `' >>kernel_code.go
	cat kernels.ptx >>kernel_code.go
	echo '`' >>kernel_code.go
	rm kernels.ptx

clean:
	rm kernel_code.go

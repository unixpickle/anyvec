package cuda

func validateGemm(tA, tB bool, m, n, k, aLen, lda, bLen, ldb, cLen, ldc int) {
	// Checking code taken from https://github.com/gonum/blas/blob/447542bc23b84f8daa32470951a57bb6713d15a1/cgo/blas.go#L2848-L2876
	// Said code is under the following license:
	//
	// Copyright ©2013 The gonum Authors. All rights reserved.
	//
	// Redistribution and use in source and binary forms, with or without
	// modification, are permitted provided that the following conditions are met:
	//     * Redistributions of source code must retain the above copyright
	//       notice, this list of conditions and the following disclaimer.
	//     * Redistributions in binary form must reproduce the above copyright
	//       notice, this list of conditions and the following disclaimer in the
	//       documentation and/or other materials provided with the distribution.
	//     * Neither the name of the gonum project nor the names of its authors and
	//       contributors may be used to endorse or promote products derived from this
	//       software without specific prior written permission.
	//
	// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if !tA {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if !tB {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > aLen || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > bLen || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > cLen || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
}

func max(i1, i2 int) int {
	if i1 > i2 {
		return i1
	}
	return i2
}

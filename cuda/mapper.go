package cuda

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/unixpickle/anyvec"
)

type mapper32 struct {
	c       *Creator32
	table   *buffer
	inSize  int
	outSize int
}

func newMapper32(c *Creator32, inSize int, table []int) (*mapper32, error) {
	if inSize > maxVector32Len {
		return nil, fmt.Errorf("input size %d too long (max is %d)", inSize,
			maxVector32Len)
	} else if len(table) > maxVector32Len {
		return nil, fmt.Errorf("input size %d too long (max is %d)", len(table),
			maxVector32Len)
	}
	for _, x := range table {
		if x >= inSize || x < 0 {
			return nil, errors.New("table index out of bounds")
		}
	}
	buf, err := newBuffer(c.handle, 4*len(table))
	if err != nil {
		return nil, err
	}
	if err := buf.SetInts(table); err != nil {
		return nil, err
	}
	return &mapper32{
		c:       c,
		table:   buf,
		inSize:  inSize,
		outSize: len(table),
	}, nil
}

func (m *mapper32) Creator() anyvec.Creator {
	return m.c
}

func (m *mapper32) InSize() int {
	return m.inSize
}

func (m *mapper32) OutSize() int {
	return m.outSize
}

func (m *mapper32) Map(in, out anyvec.Vector) {
	if in.Len() != m.inSize {
		panic("incorrect input size")
	}
	if out.Len() != m.outSize {
		panic("incorrect output size")
	}
	if in == out {
		panic("input cannot equal output")
	}
	inBuf := in.(*vector32).buffer
	outBuf := out.(*vector32).buffer
	m.c.handle.runWithKernelsAsync(func() {
		m.c.handle.kernels.MapForward32(m.outSize, m.table.ptr, outBuf.ptr, inBuf.ptr)
		runtime.KeepAlive(m.table)
		runtime.KeepAlive(inBuf)
		runtime.KeepAlive(outBuf)
	})
}

func (m *mapper32) MapTranspose(in, out anyvec.Vector) {
	if in.Len() != m.outSize {
		panic("incorrect input size")
	}
	if out.Len() != m.inSize {
		panic("incorrect output size")
	}
	if in == out {
		panic("input cannot equal output")
	}
	inBuf := in.(*vector32).buffer
	outBuf := out.(*vector32).buffer
	m.c.handle.runWithKernelsAsync(func() {
		m.c.handle.kernels.MapBackward32(m.outSize, m.table.ptr, outBuf.ptr, inBuf.ptr)
		runtime.KeepAlive(m.table)
		runtime.KeepAlive(inBuf)
		runtime.KeepAlive(outBuf)
	})
}

// Package anyvecsave provides a mechanism for serializing
// and deserializing an anyvec.Vector using the serializer
// package.
package anyvecsave

import (
	"fmt"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/unixpickle/serializer"
)

func init() {
	var s S
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeS)
}

// S is a serializer.Serializer for saving vectors.
// When deserializing, it uses creators from stardard
// sub-packages like anyvec32.
type S struct {
	Vector anyvec.Vector
}

// DeserializeS deserializes a previously-serialized S.
func DeserializeS(d []byte) (*S, error) {
	data, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	switch data := data.(type) {
	case serializer.Float32Slice:
		return &S{Vector: anyvec32.MakeVectorData([]float32(data))}, nil
	case serializer.Float64Slice:
		return &S{Vector: anyvec64.MakeVectorData([]float64(data))}, nil
	default:
		return nil, fmt.Errorf("unsupported numeric type: %T", data)
	}
}

// SerializerType returns the unique ID used to serialize
// an S using the serializer package.
func (s *S) SerializerType() string {
	return "github.com/unixpickle/anyvec/anyvecsave.S"
}

// Serialize serializes the vector, provided that the
// numeric is a supported type.
// Current supported types are: float32.
func (s *S) Serialize() ([]byte, error) {
	switch data := s.Vector.Data().(type) {
	case []float32:
		return serializer.SerializeWithType(serializer.Float32Slice(data))
	case []float64:
		return serializer.SerializeWithType(serializer.Float64Slice(data))
	default:
		return nil, fmt.Errorf("unsupported numeric type: %T", data)
	}
}

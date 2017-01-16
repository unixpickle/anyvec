package anyvec

import (
	"fmt"
	"math/rand"
)

// A ProbDist is a probability distribution.
type ProbDist int

// These are the supported probability distributions.
const (
	// Uniform random values in [0, 1).
	Uniform ProbDist = iota

	// Random values which are either 0 or 1.
	Bernoulli

	// Normally-distributed random values.
	Normal
)

// A Rander can set itself to values sampled from a
// probability distribution.
// A Rander can be passed a *rand.Rand which it may or may
// not choose to use.
// The *rand.Rand may be nil, instructing the Rander to
// use some global source of random numbers.
type Rander interface {
	Rand(ProbDist, *rand.Rand)
}

// Rand sets the vector to random values.
// If the vector does not implement Rander, a default
// implementation is used.
func Rand(v Vector, p ProbDist, gen *rand.Rand) {
	if r, ok := v.(Rander); ok {
		r.Rand(p, gen)
		return
	}
	vec := make([]float64, v.Len())
	for i := range vec {
		switch p {
		case Uniform:
			if gen != nil {
				vec[i] = gen.Float64()
			} else {
				vec[i] = rand.Float64()
			}
		case Bernoulli:
			if gen != nil {
				vec[i] = float64(gen.Intn(2))
			} else {
				vec[i] = float64(rand.Intn(2))
			}
		case Normal:
			if gen != nil {
				vec[i] = gen.NormFloat64()
			} else {
				vec[i] = rand.NormFloat64()
			}
		default:
			panic(fmt.Sprintf("unrecognized distribution: %v", p))
		}
	}
	v.SetData(v.Creator().MakeNumericList(vec))
}

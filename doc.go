// Package anyvec is a precision-agnostic vector library
// with support for hardware acceleration.
//
// Everything in anyvec is done within a context called a
// Creator. A Creator can create new vectors, concatenate
// vectors, and convert between Go numerics and vectors.
// The Creator determines the numeric type (e.g. float32
// or float64) used in vectors.
// It also determines how vectors are operated on (e.g. on
// the CPU or on the GPU).
// To get a Creator for 32-bit on-CPU vectors, see the
// anyvec32 sub-package.
//
// The Vector interface represents a mutable vector.
// Vectors may be stored anywhere and may contain any kind
// of data.
// They come with a few built-in primitive operations that
// make all sorts of things possible even when you know
// nothing about how they are stored:
//
//     var v1, v2, v3 anyvec.Vector
//     // Get the vectors somehow.
//     v1.Add(v2)
//     v2.Mul(v1)
//     v3.Set(v2)
//     v3.Sub(v1)
//     // And so on; all very abstract.
//
// Since a vector can be stored anywhere in any format, it
// is only valid to mix vectors from the same Creator.
// Vectors from different Creators are not guaranteed to
// be compatible in any way.
//
// Sometimes, one needs concrete control over vector
// components.
// The Numeric and NumericList types are abstractions for
// on-CPU numerics and slices of numerics, respectively.
// For example, anyvec32 uses float32 Numerics and
// []float32 NumericLists.
// Every Creator can create a Vector from a NumericList,
// and every Vector can produce a NumericList describing
// its contents.
// However, the type of Numeric and NumericList varies
// from Creator to Creator.
// Depending on where the Vector is stored, conversion
// to and from NumericLists may be expensive; however, it
// is always possible.
// A Creator must also provide methods to convert from
// float64 to Numeric and from []float64 to NumericList.
// This makes it possible to initialize Vectors from
// pre-determined data.
// Here is an example:
//
//     var creator anyvec.Creator
//     // Get creator here.
//     v1Data := []float64{1, 2, 3, 4}
//     v1 := creator.MakeVectorData(creator.MakeNumericList(v1Data))
//     v1.Scale(creator.MakeNumeric(-15.5))
//     fmt.Printf("Final NumericList is: %v\n", v1.Data())
//
// The anyvec package provides many non-trivial vector
// operations (e.g. Rand or AddLogs).
// People implementing a new Creator might not want to
// provide native implementations for all of these extra
// functions.
// To address this, anyvec defines an interface for every
// non-trivial vector operation.
// If a Vector does not implement the interface, then a
// default, unoptimized implementation is used.
// Many of these default implementations only support
// float32 or float64 Numerics, but some are more general.
// This makes it possible to create a minimal Creator
// implementation that meets your specialized needs while
// still supporting the entire anyvec API.
//
// The anyvec API intentionally lacks an explicit way to
// access individual components of a vector.
// In general, component-by-component manipulation can be
// inefficient, especially when the vectors are located on
// a GPU.
// The Mapper interface was introduced to fill this void.
//
// The Mapper interface is crucial when you want to
// re-arrange or sum the components of a vector.
// Essentially, a Mapper defines a one-to-many permutation
// on vector components.
// With a Mapper, one can efficiently re-arrange vector
// components even when the vectors are stored remotely.
package anyvec

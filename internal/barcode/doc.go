package barcode

// Package barcode provides a pluggable interface for barcode decoding
// with a default implementation that can be enabled via build tags.
//
// The default build has no concrete backend to avoid adding CGO or
// external dependencies implicitly. Enable the gozxing-backed decoder
// with the build tag `barcode_gozxing`.
//
// Example:
//   go build -tags=barcode_gozxing ./...


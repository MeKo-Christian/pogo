package barcode

import (
    "context"
    "image"
)

// Format represents a barcode symbology.
type Format int

const (
    FormatUnknown Format = iota
    FormatQR
    FormatDataMatrix
    FormatAztec
    FormatPDF417
    FormatCode128
    FormatCode39
    FormatEAN8
    FormatEAN13
    FormatUPCA
    FormatUPCE
    FormatITF
    FormatCodabar
)

// Options controls backend decoding behavior.
type Options struct {
    // Formats constrains the set of symbologies to search.
    Formats []Format

    // TryHarder enables more exhaustive search (slower but more robust).
    TryHarder bool

    // Multi enables multi-symbol detection in a single image.
    Multi bool

    // ROI optionally restricts decoding to a sub-rectangle of the image.
    // If zero-sized or out of bounds, backends should ignore it.
    ROI image.Rectangle

    // DPI is a hint for render scale (primarily for PDF; unused by pure image backends).
    DPI int

    // MinSize is a hint for minimum expected symbol size in pixels (backend-specific handling).
    MinSize int
}

// Point is an integer point in image coordinates.
type Point struct {
    X int
    Y int
}

// Result represents a decoded barcode.
type Result struct {
    Type       Format
    Value      string
    Points     []Point       // Corner or key points if available
    BBox       image.Rectangle // Bounding box if derivable from points
    Rotation   float64       // Degrees (clockwise); 0 if unknown
    Confidence float64       // [-1] if not provided by backend
}

// Backend is a pluggable barcode decoder implementation.
type Backend interface {
    Decode(ctx context.Context, img image.Image, opts Options) ([]Result, error)
}

// NewBackend returns the default backend implementation.
// The default build has no backend; enable specific backends via build tags.
func NewBackend() (Backend, error) { return newDefaultBackend() }


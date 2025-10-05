package barcode

import (
    "context"
    "errors"
    "image"
)

var ErrNoBackend = errors.New("barcode: no decoder backend linked; build with -tags=barcode_gozxing or configure a backend")

type defaultBackend struct{}

func newDefaultBackend() (Backend, error) { return &defaultBackend{}, nil }

func (d *defaultBackend) Decode(_ context.Context, _ image.Image, _ Options) ([]Result, error) {
    return nil, ErrNoBackend
}


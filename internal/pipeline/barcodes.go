package pipeline

import (
	"context"
	"log/slog"
)

// BarcodeConfig controls optional barcode detection.
type BarcodeConfig struct {
	Enabled   bool
	Types     []string // e.g., ["qr","ean13","code128",...]
	MinSize   int
	TryHarder bool // future use
}

// DefaultBarcodeConfig returns default barcode config (disabled).
func DefaultBarcodeConfig() BarcodeConfig { return BarcodeConfig{} }

// WithBarcodes enables barcode detection with options.
func (b *Builder) WithBarcodes(enabled bool, types []string, minSize int) *Builder {
	b.cfg.Barcode.Enabled = enabled
	if len(types) > 0 {
		b.cfg.Barcode.Types = types
	}
	if minSize >= 0 {
		b.cfg.Barcode.MinSize = minSize
	}
	return b
}

// barcodeBackend abstracts the decode method we use (to simplify tests and build tags).
type barcodeBackend interface {
	Decode(ctx context.Context, img imageLike, cfg BarcodeConfig, angle int, width, height int) ([]BarcodeResult, int64, error)
}

// imageLike is the subset we need (avoid importing image here to reduce deps for tests).
// We implement adapter in barcodes_impl.go.
type imageLike interface{}

// setupBarcode initializes the barcode backend if enabled.
func (b *Builder) setupBarcode(p *Pipeline) {
	if !b.cfg.Barcode.Enabled {
		return
	}
	dec, err := newBarcodeBackend()
	if err != nil || dec == nil {
		slog.Warn("Barcode backend unavailable; disable or build with tag", "enabled", b.cfg.Barcode.Enabled, "error", err)
		return
	}
	p.barcodeDecoder = dec
}

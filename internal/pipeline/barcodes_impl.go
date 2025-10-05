package pipeline

import (
	"context"
	"image"
	"math"
	"strings"
	"time"

	ibar "github.com/MeKo-Tech/pogo/internal/barcode"
	"github.com/disintegration/imaging"
)

// concreteBarcodeBackend implements barcodeBackend using internal/barcode.
type concreteBarcodeBackend struct{}

func newBarcodeBackend() (barcodeBackend, error) { return &concreteBarcodeBackend{}, nil }

func (c *concreteBarcodeBackend) Decode(ctx context.Context, img imageLike, cfg BarcodeConfig, angle int, width, height int) ([]BarcodeResult, int64, error) {
	im, _ := img.(image.Image)
	if im == nil {
		return nil, 0, nil
	}
	// Map types
	var formats []ibar.Format
	for _, t := range cfg.Types {
		if f, ok := parseFormat(t); ok {
			formats = append(formats, f)
		}
	}
	// Build options
	opts := ibar.Options{
		Formats:   formats,
		TryHarder: cfg.TryHarder,
		Multi:     true,
		MinSize:   cfg.MinSize,
	}
	// Create backend
	be, err := ibar.NewBackend()
	if err != nil {
		return nil, 0, err
	}
	start := time.Now()
	rs, err := be.Decode(ctx, im, opts)
	dur := time.Since(start).Nanoseconds()
	if err != nil {
		return nil, dur, err
	}
	out := mapResults(rs, 1.0)

	// Adaptive multi-scale: if nothing found and image is small or min-size hint provided, upscale and retry
	if len(out) == 0 {
		b := im.Bounds()
		minSide := b.Dx()
		if b.Dy() < minSide {
			minSide = b.Dy()
		}
		// Conditions to try upscales
		if cfg.MinSize > 0 || minSide < 600 {
			scales := []float64{1.5, 2.0}
			const maxDim = 2000
			for _, s := range scales {
				newW := int(math.Round(float64(b.Dx()) * s))
				newH := int(math.Round(float64(b.Dy()) * s))
				if newW > maxDim || newH > maxDim {
					// cap scale to fit within maxDim
					factor := float64(maxDim) / float64(max(b.Dx(), b.Dy()))
					if factor <= 1.0 {
						continue
					}
					newW = int(math.Round(float64(b.Dx()) * factor))
					newH = int(math.Round(float64(b.Dy()) * factor))
					s = factor
				}
				up := imaging.Resize(im, newW, newH, imaging.Lanczos)
				rs2, err2 := be.Decode(ctx, up, opts)
				if err2 != nil {
					continue
				}
				mapped := mapResults(rs2, s)
				if len(mapped) > 0 {
					out = mapped
					break
				}
			}
		}
	}

	return out, dur, nil
}

func mapResults(rs []ibar.Result, scale float64) []BarcodeResult {
	out := make([]BarcodeResult, 0, len(rs))
	for _, r := range rs {
		br := BarcodeResult{Type: formatToString(r.Type), Value: r.Value, Confidence: r.Confidence, Rotation: 0}
		if len(r.Points) > 0 {
			br.Points = make([]struct{ X, Y int }, 0, len(r.Points))
			// Initialize with first point (scaled back)
			minX := int(math.Round(float64(r.Points[0].X) / scale))
			minY := int(math.Round(float64(r.Points[0].Y) / scale))
			maxX, maxY := minX, minY
			for _, p := range r.Points {
				sx := int(math.Round(float64(p.X) / scale))
				sy := int(math.Round(float64(p.Y) / scale))
				br.Points = append(br.Points, struct{ X, Y int }{X: sx, Y: sy})
				if sx < minX {
					minX = sx
				}
				if sy < minY {
					minY = sy
				}
				if sx > maxX {
					maxX = sx
				}
				if sy > maxY {
					maxY = sy
				}
			}
			br.Box = struct{ X, Y, W, H int }{X: minX, Y: minY, W: maxX - minX + 1, H: maxY - minY + 1}
		}
		out = append(out, br)
	}
	return out
}

func parseFormat(s string) (ibar.Format, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "qr":
		return ibar.FormatQR, true
	case "datamatrix", "data-matrix":
		return ibar.FormatDataMatrix, true
	case "aztec":
		return ibar.FormatAztec, true
	case "pdf417":
		return ibar.FormatPDF417, true
	case "code128", "code-128":
		return ibar.FormatCode128, true
	case "code39", "code-39":
		return ibar.FormatCode39, true
	case "ean8", "ean-8":
		return ibar.FormatEAN8, true
	case "ean13", "ean-13":
		return ibar.FormatEAN13, true
	case "upca", "upc-a":
		return ibar.FormatUPCA, true
	case "upce", "upc-e":
		return ibar.FormatUPCE, true
	case "itf", "interleaved2of5", "i2/5":
		return ibar.FormatITF, true
	case "codabar":
		return ibar.FormatCodabar, true
	default:
		return 0, false
	}
}

func formatToString(f ibar.Format) string {
	switch f {
	case ibar.FormatQR:
		return "qr"
	case ibar.FormatDataMatrix:
		return "datamatrix"
	case ibar.FormatAztec:
		return "aztec"
	case ibar.FormatPDF417:
		return "pdf417"
	case ibar.FormatCode128:
		return "code128"
	case ibar.FormatCode39:
		return "code39"
	case ibar.FormatEAN8:
		return "ean8"
	case ibar.FormatEAN13:
		return "ean13"
	case ibar.FormatUPCA:
		return "upca"
	case ibar.FormatUPCE:
		return "upce"
	case ibar.FormatITF:
		return "itf"
	case ibar.FormatCodabar:
		return "codabar"
	default:
		return "unknown"
	}
}

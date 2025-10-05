package pipeline

import (
    "context"
    "image"
    "strings"
    "time"

    ibar "github.com/MeKo-Tech/pogo/internal/barcode"
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
    // Map results in working-image coordinates (overlay handles orientation transform)
    out := make([]BarcodeResult, 0, len(rs))
    for _, r := range rs {
        br := BarcodeResult{Type: formatToString(r.Type), Value: r.Value, Confidence: r.Confidence}
        if len(r.Points) > 0 {
            br.Points = make([]struct{ X, Y int }, 0, len(r.Points))
            minX, minY := r.Points[0].X, r.Points[0].Y
            maxX, maxY := minX, minY
            for _, p := range r.Points[1:] {
                br.Points = append(br.Points, struct{ X, Y int }{X: p.X, Y: p.Y})
                if p.X < minX { minX = p.X }
                if p.Y < minY { minY = p.Y }
                if p.X > maxX { maxX = p.X }
                if p.Y > maxY { maxY = p.Y }
            }
            br.Box = struct{ X, Y, W, H int }{X: minX, Y: minY, W: maxX - minX + 1, H: maxY - minY + 1}
        }
        out = append(out, br)
    }
    return out, dur, nil
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

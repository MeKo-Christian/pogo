//go:build barcode_gozxing

package barcode

import (
    "context"
    "fmt"
    "image"
    "image/draw"

    gozxing "github.com/makiuchi-d/gozxing"
    "github.com/makiuchi-d/gozxing/aztec"
    "github.com/makiuchi-d/gozxing/common"
    "github.com/makiuchi-d/gozxing/datamatrix"
    "github.com/makiuchi-d/gozxing/multi"
    "github.com/makiuchi-d/gozxing/oned"
    "github.com/makiuchi-d/gozxing/pdf417"
    "github.com/makiuchi-d/gozxing/qrcode"
)

// newDefaultBackend returns the gozxing-backed implementation when the build tag is enabled.
func newDefaultBackend() (Backend, error) { return &gozxingBackend{}, nil }

type gozxingBackend struct{}

func (b *gozxingBackend) Decode(_ context.Context, img image.Image, opts Options) ([]Result, error) {
    // Apply ROI if requested and valid
    if !opts.ROI.Empty() {
        if roiImg, ok := subImage(img, opts.ROI); ok {
            img = roiImg
        }
    }

    // Build hint map
    hints := make(map[gozxing.DecodeHintType]interface{})
    if len(opts.Formats) > 0 {
        var formats []gozxing.BarcodeFormat
        for _, f := range opts.Formats {
            if bf, ok := mapFormatToZXing(f); ok {
                formats = append(formats, bf)
            }
        }
        if len(formats) > 0 {
            hints[gozxing.DecodeHintType_POSSIBLE_FORMATS] = formats
        }
    }
    if opts.TryHarder {
        hints[gozxing.DecodeHintType_TRY_HARDER] = true
    }

    // Convert to luminance source and binary bitmap
    source := gozxing.NewLuminanceSourceFromImage(img)
    bitmap := gozxing.NewBinaryBitmap(common.NewHybridBinarizer(source))

    // Multi vs single mode
    var results []*gozxing.Result
    var err error
    if opts.Multi {
        reader := multi.NewGenericMultipleBarcodeReader(newMultiFormatReader())
        results, err = reader.DecodeMultipleWithoutHints(bitmap)
        if err != nil && len(hints) > 0 {
            // Retry with hints if available
            results, err = reader.DecodeMultiple(bitmap, hints)
        }
    } else {
        reader := newMultiFormatReader()
        var r *gozxing.Result
        r, err = reader.DecodeWithoutHints(bitmap)
        if err != nil && len(hints) > 0 {
            r, err = reader.Decode(bitmap, hints)
        }
        if err == nil && r != nil {
            results = []*gozxing.Result{r}
        }
    }
    if err != nil {
        return nil, err
    }

    // Normalize results
    out := make([]Result, 0, len(results))
    for _, r := range results {
        f := mapFormatFromZXing(r.GetBarcodeFormat())
        pts := r.GetResultPoints()
        var points []Point
        if len(pts) > 0 {
            points = make([]Point, 0, len(pts))
            for _, p := range pts {
                points = append(points, Point{X: int(p.GetX()), Y: int(p.GetY())})
            }
        }
        bbox := rectFromPoints(points)
        out = append(out, Result{
            Type:       f,
            Value:      r.GetText(),
            Points:     points,
            BBox:       bbox,
            Rotation:   0,   // Not exposed by gozxing results
            Confidence: -1,  // gozxing does not provide calibrated confidence
        })
    }
    return out, nil
}

func newMultiFormatReader() *multi.MultiFormatReader {
    // Assemble a MultiFormatReader with support for common formats.
    // The gozxing multi format reader internally consults individual format readers.
    reader := multi.NewMultiFormatReader()

    // Attach core readers (1D + 2D). Some implementations auto-configure; this is defensive.
    // Oned (1D)
    _ = oned.NewCodaBarReader()
    _ = oned.NewCode128Reader()
    _ = oned.NewCode39Reader()
    _ = oned.NewITFReader()
    _ = oned.NewEAN8Reader()
    _ = oned.NewEAN13Reader()
    _ = oned.NewUPCAReader()
    _ = oned.NewUPCEReader()

    // 2D
    _ = qrcode.NewQRCodeReader()
    _ = datamatrix.NewDataMatrixReader()
    _ = aztec.NewAztecReader()
    _ = pdf417.NewPDF417Reader()

    return reader
}

func mapFormatToZXing(f Format) (gozxing.BarcodeFormat, bool) {
    switch f {
    case FormatQR:
        return gozxing.BarcodeFormat_QR_CODE, true
    case FormatDataMatrix:
        return gozxing.BarcodeFormat_DATA_MATRIX, true
    case FormatAztec:
        return gozxing.BarcodeFormat_AZTEC, true
    case FormatPDF417:
        return gozxing.BarcodeFormat_PDF_417, true
    case FormatCode128:
        return gozxing.BarcodeFormat_CODE_128, true
    case FormatCode39:
        return gozxing.BarcodeFormat_CODE_39, true
    case FormatEAN8:
        return gozxing.BarcodeFormat_EAN_8, true
    case FormatEAN13:
        return gozxing.BarcodeFormat_EAN_13, true
    case FormatUPCA:
        return gozxing.BarcodeFormat_UPC_A, true
    case FormatUPCE:
        return gozxing.BarcodeFormat_UPC_E, true
    case FormatITF:
        return gozxing.BarcodeFormat_ITF, true
    case FormatCodabar:
        return gozxing.BarcodeFormat_CODABAR, true
    default:
        return 0, false
    }
}

func mapFormatFromZXing(bf gozxing.BarcodeFormat) Format {
    switch bf {
    case gozxing.BarcodeFormat_QR_CODE:
        return FormatQR
    case gozxing.BarcodeFormat_DATA_MATRIX:
        return FormatDataMatrix
    case gozxing.BarcodeFormat_AZTEC:
        return FormatAztec
    case gozxing.BarcodeFormat_PDF_417:
        return FormatPDF417
    case gozxing.BarcodeFormat_CODE_128:
        return FormatCode128
    case gozxing.BarcodeFormat_CODE_39:
        return FormatCode39
    case gozxing.BarcodeFormat_EAN_8:
        return FormatEAN8
    case gozxing.BarcodeFormat_EAN_13:
        return FormatEAN13
    case gozxing.BarcodeFormat_UPC_A:
        return FormatUPCA
    case gozxing.BarcodeFormat_UPC_E:
        return FormatUPCE
    case gozxing.BarcodeFormat_ITF:
        return FormatITF
    case gozxing.BarcodeFormat_CODABAR:
        return FormatCodabar
    default:
        return FormatUnknown
    }
}

func rectFromPoints(pts []Point) image.Rectangle {
    if len(pts) == 0 {
        return image.Rectangle{}
    }
    minX, minY := pts[0].X, pts[0].Y
    maxX, maxY := pts[0].X, pts[0].Y
    for _, p := range pts[1:] {
        if p.X < minX {
            minX = p.X
        }
        if p.Y < minY {
            minY = p.Y
        }
        if p.X > maxX {
            maxX = p.X
        }
        if p.Y > maxY {
            maxY = p.Y
        }
    }
    return image.Rect(minX, minY, maxX+1, maxY+1)
}

// subImage returns a sub-image if supported by the image implementation.
func subImage(img image.Image, r image.Rectangle) (image.Image, bool) {
    // Ensure ROI intersects bounds
    rb := r.Intersect(img.Bounds())
    if rb.Empty() {
        return nil, false
    }
    type subImager interface{ SubImage(r image.Rectangle) image.Image }
    if s, ok := img.(subImager); ok {
        return s.SubImage(rb), true
    }
    // Fallback: copy into new RGBA
    dst := image.NewRGBA(image.Rect(0, 0, rb.Dx(), rb.Dy()))
    draw.Draw(dst, dst.Bounds(), img, rb.Min, draw.Src)
    return dst, true
}

// Guard unused imports in case APIs vary across versions.
var _ = fmt.Sprintf


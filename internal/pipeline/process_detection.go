package pipeline

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/recognizer"
)

// performDetection runs text detection on the image.
func (p *Pipeline) performDetection(ctx context.Context, img image.Image) ([]detector.DetectedRegion, int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	slog.Debug("Starting text detection")
	detStart := time.Now()
	regions, err := p.Detector.DetectRegions(img)
	if err != nil {
		return nil, 0, fmt.Errorf("detection failed: %w", err)
	}
	detNs := time.Since(detStart).Nanoseconds()
	slog.Debug("Text detection completed", "regions_found", len(regions), "duration_ms", detNs/1000000)
	return regions, detNs, nil
}

// performRecognition runs text recognition on detected regions.
func (p *Pipeline) performRecognition(ctx context.Context, img image.Image,
	regions []detector.DetectedRegion,
) ([]recognizer.Result, int64, error) {
	recStart := time.Now()
	var recResults []recognizer.Result
	if len(regions) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		slog.Debug("Starting text recognition", "regions_count", len(regions))
		var err error
		recResults, err = p.Recognizer.RecognizeBatch(img, regions)
		if err != nil {
			return nil, 0, fmt.Errorf("recognition failed: %w", err)
		}
		slog.Debug("Text recognition completed", "duration_ms", time.Since(recStart).Nanoseconds()/1000000)
	} else {
		slog.Debug("No text regions detected, skipping recognition")
	}
	recNs := time.Since(recStart).Nanoseconds()
	return recResults, recNs, nil
}

// buildImageResult builds the final OCRImageResult from detection and recognition results.
func (p *Pipeline) buildImageResult(
	img image.Image,
	regions []detector.DetectedRegion,
	recResults []recognizer.Result,
	appliedAngle int,
	appliedConf float64,
	detNs, recNs, totalNs int64,
) *OCRImageResult {
	out := &OCRImageResult{}
	// Report original image dimensions; if rotated, transform regions back
	ob := img.Bounds()
	out.Width, out.Height = ob.Dx(), ob.Dy()
	if appliedAngle != 0 {
		out.Orientation.Angle = appliedAngle
		out.Orientation.Confidence = appliedConf
		out.Orientation.Applied = true
	}
	out.Regions = make([]OCRRegionResult, 0, len(regions))
	var detSum float64
	cleanOpts := recognizer.DefaultCleanOptions()
	if p.cfg.Recognizer.Language != "" {
		cleanOpts.Language = p.cfg.Recognizer.Language
	}

	for i, r := range regions {
		reg := p.buildRegionResult(r, recResults, i, appliedAngle, ob, cleanOpts)
		detSum += r.Confidence
		out.Regions = append(out.Regions, reg)
	}

	if len(regions) > 0 {
		out.AvgDetConf = detSum / float64(len(regions))
	}
	out.Processing.DetectionNs = detNs
	out.Processing.RecognitionNs = recNs
	out.Processing.TotalNs = totalNs

	return out
}

// buildRegionResult creates a single OCRRegionResult from detection and recognition data.
func (p *Pipeline) buildRegionResult(
	r detector.DetectedRegion,
	recResults []recognizer.Result,
	index int,
	appliedAngle int,
	originalBounds image.Rectangle,
	cleanOpts recognizer.CleanOptions,
) OCRRegionResult {
	var reg OCRRegionResult

	// Transform coordinates back to original orientation if applied
	toOriginal := func(x, y float64) (float64, float64) {
		switch appliedAngle {
		case 90:
			return float64(originalBounds.Dx()-1) - y, x
		case 180:
			return float64(originalBounds.Dx()-1) - x, float64(originalBounds.Dy()-1) - y
		case 270:
			return y, float64(originalBounds.Dy()-1) - x
		default:
			return x, y
		}
	}

	// Transform AABB by mapping its corners
	bx := float64(r.Box.MinX)
	by := float64(r.Box.MinY)
	bw := float64(r.Box.Width())
	bh := float64(r.Box.Height())
	x1, y1 := toOriginal(bx, by)
	x2, y2 := toOriginal(bx+bw, by)
	x3, y3 := toOriginal(bx+bw, by+bh)
	x4, y4 := toOriginal(bx, by+bh)
	minX := minf4(x1, x2, x3, x4)
	maxX := maxf4(x1, x2, x3, x4)
	minY := minf4(y1, y2, y3, y4)
	maxY := maxf4(y1, y2, y3, y4)
	reg.Box = struct{ X, Y, W, H int }{
		X: int(minX + 0.5),
		Y: int(minY + 0.5),
		W: int(maxX - minX + 0.5),
		H: int(maxY - minY + 0.5),
	}

	reg.Polygon = make([]struct{ X, Y float64 }, len(r.Polygon))
	for j, pt := range r.Polygon {
		ox, oy := toOriginal(pt.X, pt.Y)
		reg.Polygon[j] = struct{ X, Y float64 }{ox, oy}
	}

	reg.DetConfidence = r.Confidence

	// Add recognition results if available
	if index < len(recResults) {
		rr := recResults[index]
		text := recognizer.PostProcessText(rr.Text, cleanOpts)
		reg.Text = text
		reg.RecConfidence = rr.Confidence
		reg.CharConfidences = rr.CharConfidences
		reg.Rotated = rr.Rotated
		reg.Language = recognizer.DetectLanguage(text)
		reg.Timing.RecognizePreprocessNs = rr.TimingNs.Preprocess
		reg.Timing.RecognizeModelNs = rr.TimingNs.Model
		reg.Timing.RecognizeDecodeNs = rr.TimingNs.Decode
		reg.Timing.RecognizeTotalNs = rr.TimingNs.Total
	}

	return reg
}

func minf4(a, b, c, d float64) float64 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}

func maxf4(a, b, c, d float64) float64 {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	if d > m {
		m = d
	}
	return m
}

package detector

import (
	"image"
	"log/slog"

	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// detectRegionsMultiScale runs detection over multiple scales and merges results.
func (d *Detector) detectRegionsMultiScale(img image.Image) ([]DetectedRegion, error) {
	if img == nil {
		return nil, nil
	}

	// Get original bounds early for adaptive scale generation
	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()

	// Determine scales: adaptive or configured
	var scales []float64
	if d.config.MultiScale.Adaptive {
		scales = generateAdaptiveScales(origW, origH, d.config.MultiScale)
	} else {
		scales = d.config.MultiScale.Scales
		if len(scales) == 0 {
			scales = DefaultMultiScaleConfig().Scales
		}
	}

	var merged []DetectedRegion
	opts := PostProcessOptions{UseMinAreaRect: d.config.PolygonMode != "contour"}

	for _, s := range scales {
		if s <= 0 {
			continue
		}

		// Compute target max dimensions for this scale, clamped to minimums
		maxW := int(float64(origW) * s)
		maxH := int(float64(origH) * s)
		if maxW < 32 {
			maxW = 32
		}
		if maxH < 32 {
			maxH = 32
		}

		// Create scaled image using ResizeImage constraints to ensure multiples of 32
		scaled, err := utils.ResizeImage(img, utils.ImageConstraints{
			MaxWidth:  maxW,
			MaxHeight: maxH,
			MinWidth:  32,
			MinHeight: 32,
		})
		if err != nil {
			slog.Warn("Multi-scale resize failed, skipping scale", "scale", s, "error", err)
			continue
		}

		// Normalize without additional resizing
		tensorData, width, height, err := utils.NormalizeImagePooled(scaled)
		if err != nil {
			slog.Warn("Multi-scale normalize failed, skipping scale", "scale", s, "error", err)
			continue
		}

		// Create tensor and run inference
		tensor, err := onnx.NewImageTensor(tensorData, 3, height, width)
		if err != nil {
			mempool.PutFloat32(tensorData)
			slog.Warn("Multi-scale tensor creation failed, skipping scale", "scale", s, "error", err)
			continue
		}

		outputData, mapW, mapH, err := d.runInferenceInternal(tensor)
		// Return tensor data to pool after inference
		mempool.PutFloat32(tensor.Data)
		if err != nil {
			slog.Warn("Multi-scale inference failed, skipping scale", "scale", s, "error", err)
			continue
		}

		probMap := outputData

		// Optional morphology on the probability map per scale
		usedMorph := false
		if d.config.Morphology.Operation != MorphNone {
			probMap = ApplyMorphologicalOperation(probMap, mapW, mapH, d.config.Morphology)
			usedMorph = true
		}

		// Determine thresholds (adaptive or configured)
		dbThresh := d.config.DbThresh
		boxThresh := d.config.DbBoxThresh
		if d.config.AdaptiveThresholds.Enabled {
			adaptive := CalculateAdaptiveThresholds(probMap, mapW, mapH, d.config.AdaptiveThresholds)
			dbThresh = adaptive.DbThresh
			boxThresh = adaptive.BoxThresh
		}

		// Post-process at this scale (no final NMS yet)
		regs := PostProcessDBWithOptions(probMap, mapW, mapH, dbThresh, boxThresh, opts)
		if len(regs) == 0 {
			// return morphological buffer if allocated
			if usedMorph {
				mempool.PutFloat32(probMap)
			}
			continue
		}

		// Scale regions to original image coordinates
		regs = ScaleRegionsToOriginal(regs, mapW, mapH, origW, origH)

		// Append or incrementally merge to bound memory usage
		if d.config.MultiScale.IncrementalMerge {
			merged = append(merged, regs...)
			merged = mergeMultiScaleRegions(merged, d.config)
		} else {
			merged = append(merged, regs...)
		}

		// Return morphological buffer if allocated
		if usedMorph {
			mempool.PutFloat32(probMap)
		}
	}

	if len(merged) == 0 {
		return nil, nil
	}

	// Final merge duplicates using configured strategy
	merged = mergeMultiScaleRegions(merged, d.config)

	return merged, nil
}

// mergeMultiScaleRegions merges regions across scales using configured strategy.
func mergeMultiScaleRegions(regs []DetectedRegion, cfg Config) []DetectedRegion {
	if len(regs) == 0 {
		return regs
	}
	mergeIoU := cfg.MultiScale.MergeIoU
	if mergeIoU <= 0 {
		mergeIoU = cfg.NMSThreshold
		if mergeIoU <= 0 {
			mergeIoU = 0.3
		}
	}
	switch cfg.NMSMethod {
	case "linear", "gaussian":
		return SoftNonMaxSuppression(regs, cfg.NMSMethod, mergeIoU, cfg.SoftNMSSigma, cfg.SoftNMSThresh)
	default:
		if cfg.UseAdaptiveNMS {
			return AdaptiveNonMaxSuppression(regs, mergeIoU, cfg.AdaptiveNMSScale)
		}
		if cfg.SizeAwareNMS {
			return SizeAwareNonMaxSuppression(regs, mergeIoU, cfg.SizeNMSScaleFactor, cfg.MinRegionSize, cfg.MaxRegionSize)
		}
		return NonMaxSuppression(regs, mergeIoU)
	}
}

// generateAdaptiveScales creates a set of scales based on image size and config.
// Always includes 1.0 as the first level. Subsequent levels downscale by ~0.75
// until reaching MaxLevels or when min(image side * scale) <= MinSide.
func generateAdaptiveScales(origW, origH int, ms MultiScaleConfig) []float64 {
	if origW <= 0 || origH <= 0 {
		return []float64{1.0}
	}
	// Seed with 1.0
	scales := []float64{1.0}
	// Use a default decay factor similar to common pyramids
	const factor = 0.75
	minSide := origW
	if origH < minSide {
		minSide = origH
	}
	// If existing explicit scales provided, prefer them but trim by limits
	if len(ms.Scales) > 0 {
		// Ensure 1.0 first and unique
		seen := map[float64]bool{1.0: true}
		for _, s := range ms.Scales {
			if s <= 0 {
				continue
			}
			if !seen[s] {
				scales = append(scales, s)
				seen[s] = true
			}
			if ms.MaxLevels > 0 && len(scales) >= ms.MaxLevels {
				break
			}
			if int(float64(minSide)*s) <= ms.MinSide {
				break
			}
		}
		return scales
	}

	// Generate by factor
	maxLevels := ms.MaxLevels
	if maxLevels <= 0 {
		maxLevels = 3
	}
	for len(scales) < maxLevels {
		next := scales[len(scales)-1] * factor
		if next <= 0 {
			break
		}
		// Stop if min dimension would drop below threshold
		if int(float64(minSide)*next) <= ms.MinSide {
			break
		}
		scales = append(scales, next)
	}
	return scales
}

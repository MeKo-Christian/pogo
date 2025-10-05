package detector

import (
	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// DetectedRegion represents a post-processed text region.
type DetectedRegion struct {
	// Polygon in probability-map coordinates (before scaling to original).
	Polygon []utils.Point
	// Bounding box in probability-map coordinates.
	Box utils.Box
	// Average probability across the region's pixels.
	Confidence float64
}

// PostProcessOptions controls optional behaviors in DB post-processing.
type PostProcessOptions struct {
	UseMinAreaRect bool
	// Confidence calculation and calibration
	ConfidenceMethod string  // "mean" (default), "max", or "mean_var"
	CalibrationGamma float64 // >0, 1.0 means no change; >1 down-weights, <1 up-weights
	// Adaptive confidence thresholding
	AdaptiveConfidence      bool    // reduce threshold for very small regions
	AdaptiveConfidenceScale float64 // max fractional reduction (0..1), default ~0.2
}

// PostProcessDB executes a simplified DB-style post-process:
// 1) Threshold to binary mask
// 2) 4-connected components
// 3) Compute average probability, bounding boxes, and polygons
// 4) Filter by min box confidence.
// Uses memory pooling to reduce allocations.
func PostProcessDB(prob []float32, w, h int, dbThresh, boxMinConf float32) []DetectedRegion {
	if len(prob) != w*h || w <= 0 || h <= 0 {
		return nil
	}
	mask := binarize(prob, w, h, dbThresh)
	defer mempool.PutBool(mask)

	comps, labels := connectedComponents(mask, prob, w, h)
	// Default behavior: use minimum-area rectangle for polygons
	regions := regionsFromComponents(comps, labels, w, h, PostProcessOptions{UseMinAreaRect: true})
	regions = filterRegions(regions, float64(boxMinConf))
	return regions
}

// PostProcessDBWithNMS applies DB post-processing followed by NMS.
func PostProcessDBWithNMS(prob []float32, w, h int, dbThresh, boxMinConf float32,
	iouThreshold float64,
) []DetectedRegion {
	regs := PostProcessDB(prob, w, h, dbThresh, boxMinConf)
	if len(regs) == 0 {
		return regs
	}
	return NonMaxSuppression(regs, iouThreshold)
}

// PostProcessDBWithOptions is like PostProcessDB but allows selecting polygon mode.
// Uses memory pooling to reduce allocations.
func PostProcessDBWithOptions(prob []float32, w, h int, dbThresh, boxMinConf float32,
	opts PostProcessOptions,
) []DetectedRegion {
	if len(prob) != w*h || w <= 0 || h <= 0 {
		return nil
	}
	mask := binarize(prob, w, h, dbThresh)
	defer mempool.PutBool(mask)

	comps, labels := connectedComponents(mask, prob, w, h)
	regions := regionsFromComponents(comps, labels, w, h, opts)
	// Apply (optional) adaptive confidence thresholding
	if opts.AdaptiveConfidence {
		regions = adaptiveFilterRegions(regions, float64(boxMinConf), w, h, opts)
	} else {
		regions = filterRegions(regions, float64(boxMinConf))
	}
	return regions
}

// PostProcessDBWithNMSOptions applies DB post-processing with options followed by NMS.
func PostProcessDBWithNMSOptions(prob []float32, w, h int, dbThresh, boxMinConf float32,
	iouThreshold float64, opts PostProcessOptions,
) []DetectedRegion {
	regs := PostProcessDBWithOptions(prob, w, h, dbThresh, boxMinConf, opts)
	if len(regs) == 0 {
		return regs
	}
	return NonMaxSuppression(regs, iouThreshold)
}

// ScaleRegionsToOriginal scales detected regions from probability map coordinates to original image coordinates.
func ScaleRegionsToOriginal(regions []DetectedRegion, mapW, mapH, origW, origH int) []DetectedRegion {
	if mapW == 0 || mapH == 0 {
		return regions
	}
	sx := float64(origW) / float64(mapW)
	sy := float64(origH) / float64(mapH)
	out := make([]DetectedRegion, len(regions))
	for i, r := range regions {
		// Scale polygon
		scaledPoly := make([]utils.Point, len(r.Polygon))
		for j, p := range r.Polygon {
			scaledPoly[j] = utils.Point{X: p.X * sx, Y: p.Y * sy}
		}
		// Scale box
		b := r.Box
		sb := utils.NewBox(b.MinX*sx, b.MinY*sy, b.MaxX*sx, b.MaxY*sy)
		out[i] = DetectedRegion{Polygon: scaledPoly, Box: sb, Confidence: r.Confidence}
	}
	return out
}

// adaptiveFilterRegions applies area-aware threshold adjustments to favor small regions.
func adaptiveFilterRegions(regions []DetectedRegion, baseThreshold float64, mapW, mapH int, opts PostProcessOptions) []DetectedRegion {
	scale := opts.AdaptiveConfidenceScale
	if scale <= 0 {
		scale = 0.2
	}
	if scale > 1 {
		scale = 1
	}
	totalArea := float64(mapW * mapH)
	filtered := make([]DetectedRegion, 0, len(regions))
	for _, r := range regions {
		// Compute normalized area of region's box
		bw := r.Box.MaxX - r.Box.MinX
		bh := r.Box.MaxY - r.Box.MinY
		if bw < 0 || bh < 0 {
			continue
		}
		area := float64(bw * bh)
		if area < 0 {
			continue
		}
		normArea := 0.0
		if totalArea > 0 {
			normArea = area / totalArea
		}
		// Reduce threshold up to "scale" for very small regions (<1% area), linearly
		reduction := 0.0
		smallRef := 0.01
		if normArea < smallRef {
			reduction = scale * (1 - (normArea / smallRef))
			if reduction > scale {
				reduction = scale
			} else if reduction < 0 {
				reduction = 0
			}
		}
		thr := baseThreshold * (1 - reduction)
		if r.Confidence >= thr {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

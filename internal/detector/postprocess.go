package detector

import (
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
}

// PostProcessDB executes a simplified DB-style post-process:
// 1) Threshold to binary mask
// 2) 4-connected components
// 3) Compute average probability, bounding boxes, and polygons
// 4) Filter by min box confidence.
func PostProcessDB(prob []float32, w, h int, dbThresh, boxMinConf float32) []DetectedRegion {
	if len(prob) != w*h || w <= 0 || h <= 0 {
		return nil
	}
	mask := binarize(prob, w, h, dbThresh)
	comps, labels := connectedComponents(mask, prob, w, h)
	// Default behavior: use minimum-area rectangle for polygons
	regions := regionsFromComponents(comps, labels, w, h, true)
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
func PostProcessDBWithOptions(prob []float32, w, h int, dbThresh, boxMinConf float32,
	opts PostProcessOptions,
) []DetectedRegion {
	if len(prob) != w*h || w <= 0 || h <= 0 {
		return nil
	}
	mask := binarize(prob, w, h, dbThresh)
	comps, labels := connectedComponents(mask, prob, w, h)
	regions := regionsFromComponents(comps, labels, w, h, opts.UseMinAreaRect)
	regions = filterRegions(regions, float64(boxMinConf))
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

package detector

import (
	"container/list"
	"math"
	"strings"

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

// binarize creates a binary mask from a probability map with threshold t.
func binarize(prob []float32, w, h int, t float32) []bool {
	mask := make([]bool, w*h)
	for i, p := range prob {
		if p >= t {
			mask[i] = true
		}
	}
	return mask
}

// connectedComponents finds 4-connected components in the mask and returns
// for each component its pixel indices and simple stats.
type compStats struct {
	count int
	sum   float64
	minX  int
	minY  int
	maxX  int
	maxY  int
}

// performComponentBFS performs BFS traversal for a connected component starting from a seed pixel.
func performComponentBFS(mask []bool, prob []float32, visited []int, labels []int,
	w, h, startX, startY, label int,
) compStats {
	idx := func(x, y int) int { return y*w + x }
	startIdx := idx(startX, startY)

	st := compStats{count: 0, sum: 0, minX: startX, minY: startY, maxX: startX, maxY: startY}
	q := list.New()
	q.PushBack(startIdx)
	visited[startIdx] = 1
	labels[startIdx] = label

	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}

	for q.Len() > 0 {
		e := q.Front()
		q.Remove(e)
		ci, ok := e.Value.(int)
		if !ok {
			continue // skip invalid
		}
		cx, cy := ci%w, ci/w
		updateComponentStats(&st, prob[ci], cx, cy)
		processNeighbors(mask, visited, labels, q, w, h, cx, cy, label, idx, dirs)
	}
	return st
}

// updateComponentStats updates the component statistics with a new pixel.
func updateComponentStats(st *compStats, prob float32, cx, cy int) {
	st.count++
	st.sum += float64(prob)
	if cx < st.minX {
		st.minX = cx
	}
	if cy < st.minY {
		st.minY = cy
	}
	if cx > st.maxX {
		st.maxX = cx
	}
	if cy > st.maxY {
		st.maxY = cy
	}
}

// processNeighbors processes all 4-connected neighbors of a pixel.
func processNeighbors(mask []bool, visited []int, labels []int, q *list.List,
	w, h, cx, cy, label int, idx func(int, int) int, dirs [][2]int,
) {
	for _, d := range dirs {
		nx, ny := cx+d[0], cy+d[1]
		if isValidNeighbor(mask, visited, w, h, nx, ny) {
			ni := idx(nx, ny)
			visited[ni] = 1
			labels[ni] = label
			q.PushBack(ni)
		}
	}
}

// isValidNeighbor checks if a neighbor pixel is valid for BFS traversal.
func isValidNeighbor(mask []bool, visited []int, w, h, nx, ny int) bool {
	if nx < 0 || ny < 0 || nx >= w || ny >= h {
		return false
	}
	ni := ny*w + nx
	return mask[ni] && visited[ni] == 0
}

func connectedComponents(mask []bool, prob []float32, w, h int) ([]compStats, []int) {
	visited := make([]int, len(mask)) // Use int instead of bool for clarity
	labels := make([]int, len(mask))
	var comps []compStats
	lab := 0
	idx := func(x, y int) int { return y*w + x }

	for y := range h {
		for x := range w {
			i := idx(x, y)
			if !mask[i] || visited[i] != 0 {
				continue
			}
			lab++
			st := performComponentBFS(mask, prob, visited, labels, w, h, x, y, lab)
			comps = append(comps, st)
		}
	}
	return comps, labels
}

// regionsFromComponents converts components into DetectedRegion records using
// bounding boxes as simple polygons.
func regionsFromComponents(comps []compStats, labels []int, w, h int, useMinAreaRect bool) []DetectedRegion {
	regions := make([]DetectedRegion, 0, len(comps))
	for i, c := range comps {
		if c.count == 0 {
			continue
		}
		conf := c.sum / float64(c.count)
		label := i + 1
		poly := traceContourMoore(labels, w, h, label, c)
		if len(poly) == 0 {
			// Fallback to AABB polygon
			box := utils.NewBox(float64(c.minX), float64(c.minY), float64(c.maxX+1), float64(c.maxY+1))
			poly = []utils.Point{
				{X: box.MinX, Y: box.MinY},
				{X: box.MaxX, Y: box.MinY},
				{X: box.MaxX, Y: box.MaxY},
				{X: box.MinX, Y: box.MaxY},
			}
			regions = append(regions, DetectedRegion{Polygon: poly, Box: box, Confidence: conf})
			continue
		}
		// Optional enhancements: simplify contour and unclip/expand slightly
		// Choose epsilon relative to component size to be robust across scales
		compW := float64(c.maxX - c.minX + 1)
		compH := float64(c.maxY - c.minY + 1)
		maxDim := compW
		if compH > maxDim {
			maxDim = compH
		}
		epsilon := 0.01 * maxDim // 1% of component size
		if epsilon < 0.5 {
			epsilon = 0.5
		}
		poly = utils.SimplifyPolygon(poly, epsilon)
		// Unclip by 10% outward around centroid
		poly = utils.UnclipPolygon(poly, 1.10)
		// Optionally replace polygon with its minimum-area enclosing rectangle
		if useMinAreaRect && len(poly) >= 3 {
			if mar := utils.MinimumAreaRectangle(poly); len(mar) == 4 {
				poly = mar
			}
		}
		// Compute bounding box from polygon points (treated as pixel centers)
		// Convert to pixel-edge aligned box by expanding max by +1 and clamp to map bounds
		bb := utils.BoundingBox(poly)
		minX := math.Max(0, bb.MinX)
		minY := math.Max(0, bb.MinY)
		maxX := math.Min(float64(w), bb.MaxX+1)
		maxY := math.Min(float64(h), bb.MaxY+1)
		box := utils.NewBox(minX, minY, maxX, maxY)
		regions = append(regions, DetectedRegion{Polygon: poly, Box: box, Confidence: conf})
	}
	return regions
}

// filterRegions drops regions whose confidence is below the given threshold.
func filterRegions(regions []DetectedRegion, minConf float64) []DetectedRegion {
	out := regions[:0]
	for _, r := range regions {
		if r.Confidence >= minConf {
			out = append(out, r)
		}
	}
	return out
}

// sortRegionsByConfidence sorts region indices by confidence in descending order.
func sortRegionsByConfidence(regions []DetectedRegion) []int {
	idx := make([]int, len(regions))
	for i := range idx {
		idx[i] = i
	}
	// simple selection sort (small region counts expected)
	for i := range len(idx) - 1 {
		maxJ := i
		for j := i + 1; j < len(idx); j++ {
			if regions[idx[j]].Confidence > regions[idx[maxJ]].Confidence {
				maxJ = j
			}
		}
		idx[i], idx[maxJ] = idx[maxJ], idx[i]
	}
	return idx
}

// NonMaxSuppression performs greedy NMS on regions' axis-aligned boxes.
// Keeps higher-confidence regions when IoU exceeds threshold.
func NonMaxSuppression(regions []DetectedRegion, iouThreshold float64) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	idx := sortRegionsByConfidence(regions)
	kept := make([]DetectedRegion, 0, len(regions))
	suppressed := make([]bool, len(regions))

	for i := range idx {
		a := idx[i]
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])
		for j := i + 1; j < len(idx); j++ {
			b := idx[j]
			if suppressed[b] {
				continue
			}
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > iouThreshold {
				suppressed[b] = true
			}
		}
	}
	return kept
}

// AdaptiveNonMaxSuppression performs greedy NMS with adaptive IoU thresholds.
func AdaptiveNonMaxSuppression(regions []DetectedRegion, baseThreshold, scaleFactor float64) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	idx := sortRegionsByConfidence(regions)
	kept := make([]DetectedRegion, 0, len(regions))
	suppressed := make([]bool, len(regions))

	for i := range idx {
		a := idx[i]
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])
		for j := i + 1; j < len(idx); j++ {
			b := idx[j]
			if suppressed[b] {
				continue
			}
			// Use adaptive threshold based on region characteristics
			adaptiveThreshold := calculateAdaptiveIoUThreshold(baseThreshold, scaleFactor, regions[a], regions[b])
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > adaptiveThreshold {
				suppressed[b] = true
			}
		}
	}
	return kept
}

// SizeAwareNonMaxSuppression performs greedy NMS with size-based adaptive thresholds.
func SizeAwareNonMaxSuppression(regions []DetectedRegion, baseThreshold, sizeScaleFactor float64,
	minSize, maxSize int,
) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}

	idx := sortRegionsByConfidence(regions)
	kept := make([]DetectedRegion, 0, len(regions))
	suppressed := make([]bool, len(regions))

	for i := range idx {
		a := idx[i]
		if suppressed[a] {
			continue
		}
		kept = append(kept, regions[a])
		for j := i + 1; j < len(idx); j++ {
			b := idx[j]
			if suppressed[b] {
				continue
			}
			// Use size-based adaptive threshold
			adaptiveThreshold := calculateSizeBasedIoUThreshold(baseThreshold, sizeScaleFactor,
				minSize, maxSize, regions[a], regions[b])
			if ComputeRegionIoU(regions[a].Box, regions[b].Box) > adaptiveThreshold {
				suppressed[b] = true
			}
		}
	}
	return kept
}

// sortRegionsByConfidenceDesc sorts regions by confidence in descending order using selection sort.
func sortRegionsByConfidenceDesc(regions []DetectedRegion) {
	n := len(regions)
	for i := range n - 1 {
		maxJ := i
		for j := i + 1; j < n; j++ {
			if regions[j].Confidence > regions[maxJ].Confidence {
				maxJ = j
			}
		}
		regions[i], regions[maxJ] = regions[maxJ], regions[i]
	}
}

// sortRegionsByConfidenceDescFrom sorts regions starting from a given index by confidence in descending order.
func sortRegionsByConfidenceDescFrom(regions []DetectedRegion, start int) {
	n := len(regions)
	for i := start; i < n-1; i++ {
		maxJ := i
		for j := i + 1; j < n; j++ {
			if regions[j].Confidence > regions[maxJ].Confidence {
				maxJ = j
			}
		}
		regions[i], regions[maxJ] = regions[maxJ], regions[i]
	}
}

// calculateSoftNMSWeight computes the decay weight for Soft-NMS based on IoU and method.
func calculateSoftNMSWeight(iou, iouThreshold, sigma float64, method string) float64 {
	switch strings.ToLower(method) {
	case "linear":
		if iou > iouThreshold {
			return 1.0 - iou
		}
		return 1.0
	case "gaussian":
		if sigma <= 0 {
			sigma = 0.5
		}
		return math.Exp(-(iou * iou) / sigma)
	default:
		if iou > iouThreshold {
			return 0.0
		}
		return 1.0
	}
}

// SoftNonMaxSuppression applies Soft-NMS to a set of regions. The method can be
// "linear" or "gaussian" (case-insensitive). Boxes are decayed rather than
// suppressed. Boxes with final confidence below scoreThresh are discarded.
// Returns regions sorted by confidence descending.
func SoftNonMaxSuppression(regions []DetectedRegion, method string,
	iouThreshold, sigma, scoreThresh float64,
) []DetectedRegion {
	n := len(regions)
	if n <= 1 {
		return handleEdgeCases(regions, n, scoreThresh)
	}

	// Work on a copy to avoid mutating input
	regs := make([]DetectedRegion, n)
	copy(regs, regions)

	// Initial sort
	sortRegionsByConfidenceDesc(regs)

	// Soft-NMS loop
	applySoftNMS(&regs, iouThreshold, sigma, method, scoreThresh)

	// Filter by score threshold and final sort
	return filterAndSortResults(regs, scoreThresh)
}

// handleEdgeCases handles the edge cases for Soft-NMS when there are 0 or 1 regions.
func handleEdgeCases(regions []DetectedRegion, n int, scoreThresh float64) []DetectedRegion {
	if n == 1 && regions[0].Confidence < scoreThresh {
		return nil
	}
	return regions
}

// applySoftNMS applies the Soft-NMS algorithm to the regions.
func applySoftNMS(regs *[]DetectedRegion, iouThreshold, sigma float64, method string, scoreThresh float64) {
	n := len(*regs)
	for i := range n {
		// Ensure regs[i] has highest score among remaining by resorting tail
		if i > 0 {
			sortRegionsByConfidenceDescFrom(*regs, i)
		}
		decayOverlappingRegions(*regs, i, iouThreshold, sigma, method, scoreThresh)
	}
}

// decayOverlappingRegions decays the confidence of regions that overlap with the current region.
func decayOverlappingRegions(regs []DetectedRegion, i int,
	iouThreshold, sigma float64, method string, scoreThresh float64,
) {
	for j := i + 1; j < len(regs); j++ {
		if regs[j].Confidence < scoreThresh {
			continue
		}
		iou := ComputeRegionIoU(regs[i].Box, regs[j].Box)
		if iou <= 0 {
			continue
		}
		weight := calculateSoftNMSWeight(iou, iouThreshold, sigma, method)
		regs[j].Confidence *= weight
	}
}

// filterAndSortResults filters regions by score threshold and sorts the final results.
func filterAndSortResults(regs []DetectedRegion, scoreThresh float64) []DetectedRegion {
	out := make([]DetectedRegion, 0, len(regs))
	for _, r := range regs {
		if r.Confidence >= scoreThresh {
			out = append(out, r)
		}
	}
	if len(out) <= 1 {
		return out
	}
	// Final sort by confidence desc
	sortRegionsByConfidenceDesc(out)
	return out
}

// ScaleRegionsToOriginal maps probability-map coordinates back to original
// image size using linear scaling.
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

// PostProcessOptions controls optional behaviors in DB post-processing.
type PostProcessOptions struct {
	UseMinAreaRect bool
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

// ComputeRegionIoU computes IoU of two axis-aligned boxes for testing/filtering.
func ComputeRegionIoU(a, b utils.Box) float64 {
	ix1 := math.Max(a.MinX, b.MinX)
	iy1 := math.Max(a.MinY, b.MinY)
	ix2 := math.Min(a.MaxX, b.MaxX)
	iy2 := math.Min(a.MaxY, b.MaxY)
	iw := math.Max(0, ix2-ix1)
	ih := math.Max(0, iy2-iy1)
	inter := iw * ih
	if inter <= 0 {
		return 0
	}
	aArea := a.Width() * a.Height()
	bArea := b.Width() * b.Height()
	return inter / (aArea + bArea - inter)
}

// AdaptiveNMSThresholds contains tunable parameters for adaptive NMS.
type AdaptiveNMSThresholds struct {
	BaseThreshold float64 // Base IoU threshold
	ScaleFactor   float64 // Overall scaling factor
	SizeWeight    float64 // Weight for size-based adjustment
	ConfWeight    float64 // Weight for confidence-based adjustment
	MinThreshold  float64 // Minimum allowed threshold
	MaxThreshold  float64 // Maximum allowed threshold
}

// DefaultAdaptiveNMSThresholds returns default adaptive NMS parameters.
func DefaultAdaptiveNMSThresholds() AdaptiveNMSThresholds {
	return AdaptiveNMSThresholds{
		BaseThreshold: 0.3,
		ScaleFactor:   1.0,
		SizeWeight:    0.1,
		ConfWeight:    0.05,
		MinThreshold:  0.1,
		MaxThreshold:  0.8,
	}
}

// calculateAdaptiveIoUThreshold computes an adaptive IoU threshold based on region characteristics.
func calculateAdaptiveIoUThreshold(baseThreshold, scaleFactor float64, regionA, regionB DetectedRegion) float64 {
	// Calculate size-based adjustment
	sizeA := regionA.Box.Width() * regionA.Box.Height()
	sizeB := regionB.Box.Width() * regionB.Box.Height()
	avgSize := (sizeA + sizeB) / 2.0

	// Normalize size (assuming typical text region sizes)
	normalizedSize := math.Min(avgSize/10000.0, 1.0) // Cap at 1.0 for very large regions

	// Calculate confidence-based adjustment
	avgConf := (regionA.Confidence + regionB.Confidence) / 2.0

	// Adaptive threshold: base + size adjustment + confidence adjustment
	adaptiveThreshold := baseThreshold * scaleFactor
	adaptiveThreshold += 0.1 * normalizedSize   // Larger regions can have higher IoU tolerance
	adaptiveThreshold -= 0.05 * (avgConf - 0.5) // Higher confidence regions can be more strict

	// Clamp to reasonable bounds
	if adaptiveThreshold < 0.1 {
		adaptiveThreshold = 0.1
	}
	if adaptiveThreshold > 0.8 {
		adaptiveThreshold = 0.8
	}

	return adaptiveThreshold
}

// calculateSizeBasedIoUThreshold computes size-aware IoU threshold.
func calculateSizeBasedIoUThreshold(baseThreshold, sizeScaleFactor float64,
	minSize, maxSize int, regionA, regionB DetectedRegion,
) float64 {
	sizeA := regionA.Box.Width() * regionA.Box.Height()
	sizeB := regionB.Box.Width() * regionB.Box.Height()
	avgSize := (sizeA + sizeB) / 2.0

	// Normalize size between min and max
	sizeRange := float64(maxSize - minSize)
	if sizeRange <= 0 {
		return baseThreshold
	}

	normalizedSize := (avgSize - float64(minSize)) / sizeRange
	normalizedSize = math.Max(0, math.Min(1, normalizedSize)) // Clamp to [0,1]

	// Smaller regions get stricter thresholds (less tolerance for overlap)
	// Larger regions get more lenient thresholds (more tolerance for overlap)
	sizeAdjustment := sizeScaleFactor * (normalizedSize - 0.5)

	return baseThreshold + sizeAdjustment
}

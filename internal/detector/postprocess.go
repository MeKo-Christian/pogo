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

func connectedComponents(mask []bool, prob []float32, w, h int) ([]compStats, []int) {
	visited := make([]bool, len(mask))
	labels := make([]int, len(mask))
	var comps []compStats
	lab := 0
	idx := func(x, y int) int { return y*w + x }

	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for y := range h {
		for x := range w {
			i := idx(x, y)
			if !mask[i] || visited[i] {
				continue
			}
			// BFS
			st := compStats{count: 0, sum: 0, minX: x, minY: y, maxX: x, maxY: y}
			q := list.New()
			q.PushBack(i)
			visited[i] = true
			lab++
			labels[i] = lab
			for q.Len() > 0 {
				e := q.Front()
				q.Remove(e)
				ci := e.Value.(int)
				cx, cy := ci%w, ci/w
				st.count++
				st.sum += float64(prob[ci])
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
				for _, d := range dirs {
					nx, ny := cx+d[0], cy+d[1]
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					ni := idx(nx, ny)
					if mask[ni] && !visited[ni] {
						visited[ni] = true
						labels[ni] = lab
						q.PushBack(ni)
					}
				}
			}
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

// NonMaxSuppression performs greedy NMS on regions' axis-aligned boxes.
// Keeps higher-confidence regions when IoU exceeds threshold.
func NonMaxSuppression(regions []DetectedRegion, iouThreshold float64) []DetectedRegion {
	if len(regions) <= 1 {
		return regions
	}
	// Sort indices by confidence desc
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

// SoftNonMaxSuppression applies Soft-NMS to a set of regions. The method can be
// "linear" or "gaussian" (case-insensitive). Boxes are decayed rather than
// suppressed. Boxes with final confidence below scoreThresh are discarded.
// Returns regions sorted by confidence descending.
func SoftNonMaxSuppression(regions []DetectedRegion, method string, iouThreshold, sigma, scoreThresh float64) []DetectedRegion {
	n := len(regions)
	if n <= 1 {
		if n == 1 && regions[0].Confidence < scoreThresh {
			return nil
		}
		return regions
	}
	// Work on a copy to avoid mutating input
	regs := make([]DetectedRegion, n)
	copy(regs, regions)

	// Helper for sorting by confidence desc
	sortIdx := func(start int) {
		// selection sort on the tail; small n expected
		for i := start; i < n-1; i++ {
			maxJ := i
			for j := i + 1; j < n; j++ {
				if regs[j].Confidence > regs[maxJ].Confidence {
					maxJ = j
				}
			}
			regs[i], regs[maxJ] = regs[maxJ], regs[i]
		}
	}

	// Initial sort
	sortIdx(0)

	// Soft-NMS loop
	for i := range n {
		// Ensure regs[i] has highest score among remaining by resorting tail
		if i > 0 {
			// resort tail from i
			// simple selection sort from i
			maxJ := i
			for j := i + 1; j < n; j++ {
				if regs[j].Confidence > regs[maxJ].Confidence {
					maxJ = j
				}
			}
			regs[i], regs[maxJ] = regs[maxJ], regs[i]
		}
		for j := i + 1; j < n; j++ {
			if regs[j].Confidence < scoreThresh {
				continue
			}
			iou := ComputeRegionIoU(regs[i].Box, regs[j].Box)
			if iou <= 0 {
				continue
			}
			// compute decay weight
			weight := 1.0
			switch strings.ToLower(method) {
			case "linear":
				if iou > iouThreshold {
					weight = 1.0 - iou
				}
			case "gaussian":
				if sigma <= 0 {
					sigma = 0.5
				}
				weight = math.Exp(-(iou * iou) / sigma)
			default:
				if iou > iouThreshold {
					weight = 0.0
				}
			}
			regs[j].Confidence *= weight
		}
	}

	// Filter by score threshold and sort
	out := make([]DetectedRegion, 0, n)
	for _, r := range regs {
		if r.Confidence >= scoreThresh {
			out = append(out, r)
		}
	}
	if len(out) <= 1 {
		return out
	}
	// Final sort by confidence desc
	for i := range len(out) - 1 {
		maxJ := i
		for j := i + 1; j < len(out); j++ {
			if out[j].Confidence > out[maxJ].Confidence {
				maxJ = j
			}
		}
		out[i], out[maxJ] = out[maxJ], out[i]
	}
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
func PostProcessDBWithNMS(prob []float32, w, h int, dbThresh, boxMinConf float32, iouThreshold float64) []DetectedRegion {
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
func PostProcessDBWithOptions(prob []float32, w, h int, dbThresh, boxMinConf float32, opts PostProcessOptions) []DetectedRegion {
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
func PostProcessDBWithNMSOptions(prob []float32, w, h int, dbThresh, boxMinConf float32, iouThreshold float64, opts PostProcessOptions) []DetectedRegion {
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

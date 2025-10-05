package detector

import (
	"container/list"
	"math"

	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/MeKo-Tech/pogo/internal/utils"
)

// compStats represents statistics for a connected component.
type compStats struct {
	count int
	sum   float64
	sumSq float64
	maxV  float64
	minX  int
	minY  int
	maxX  int
	maxY  int
}

// binarize creates a binary mask from a probability map with threshold t.
// Uses memory pooling for the boolean mask.
func binarize(prob []float32, w, h int, t float32) []bool {
	mask := mempool.GetBool(w * h)
	for i, p := range prob {
		if p >= t {
			mask[i] = true
		}
	}
	return mask
}

// connectedComponents finds 4-connected components in the mask and returns
// for each component its pixel indices and simple stats.
func connectedComponents(mask []bool, prob []float32, w, h int) ([]compStats, []int) {
	visited := make([]int, w*h)
	labels := make([]int, w*h)
	var comps []compStats
	label := 1

	for y := range h {
		for x := range w {
			idx := y*w + x
			if mask[idx] && visited[idx] == 0 {
				st := performComponentBFS(mask, prob, visited, labels, w, h, x, y, label)
				comps = append(comps, st)
				label++
			}
		}
	}

	return comps, labels
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
	st.sumSq += float64(prob) * float64(prob)
	if float64(prob) > st.maxV {
		st.maxV = float64(prob)
	}
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

// isValidNeighbor checks if a neighbor pixel is valid for component expansion.
func isValidNeighbor(mask []bool, visited []int, w, h, nx, ny int) bool {
	return nx >= 0 && nx < w && ny >= 0 && ny < h &&
		mask[ny*w+nx] && visited[ny*w+nx] == 0
}

// regionsFromComponents converts connected components to detected regions.
func regionsFromComponents(comps []compStats, labels []int, w, h int, opts PostProcessOptions) []DetectedRegion {
	regions := make([]DetectedRegion, 0, len(comps))
	for i, c := range comps {
		if c.count == 0 {
			continue
		}
		conf := computeComponentConfidence(c, opts)
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
		if opts.UseMinAreaRect && len(poly) >= 3 {
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

// computeComponentConfidence computes confidence for a component based on options.
// Supported methods:
// - "mean" (default): average probability
// - "max": maximum probability within the component
// - "mean_var": variance-adjusted mean (penalize high-variance regions).
func computeComponentConfidence(c compStats, opts PostProcessOptions) float64 {
	// Default to mean
	method := opts.ConfidenceMethod
	if method == "" {
		method = "mean"
	}
	mean := 0.0
	if c.count > 0 {
		mean = c.sum / float64(c.count)
	}

	var conf float64
	switch method {
	case "max":
		conf = c.maxV
	case "mean_var":
		// Compute unbiased variance estimate if possible
		var variance float64
		if c.count > 1 {
			m2 := c.sumSq/float64(c.count) - mean*mean
			if m2 < 0 {
				m2 = 0 // numeric guard
			}
			variance = m2
		} else {
			variance = 0
		}
		// Normalize variance by theoretical Bernoulli variance bound mean*(1-mean)
		denom := mean * (1 - mean)
		normVar := 0.0
		if denom > 1e-6 {
			normVar = variance / denom
		}
		if normVar > 1 {
			normVar = 1
		} else if normVar < 0 {
			normVar = 0
		}
		// Penalize mean by normalized variance (up to 50% reduction)
		conf = mean * (1 - 0.5*normVar)
	case "mean":
		fallthrough
	default:
		conf = mean
	}

	// Apply simple calibration via gamma/power scaling if requested
	gamma := opts.CalibrationGamma
	if gamma > 0 && gamma != 1.0 {
		// clamp to [0,1] before and after
		if conf < 0 {
			conf = 0
		} else if conf > 1 {
			conf = 1
		}
		conf = math.Pow(conf, gamma)
	}
	// Final clamp to [0,1]
	if conf < 0 {
		conf = 0
	} else if conf > 1 {
		conf = 1
	}
	return conf
}

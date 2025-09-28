package rectify

import (
	"math"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// processMaskAndFindRectangle extracts mask points and finds the minimum area rectangle.
func (r *Rectifier) processMaskAndFindRectangle(outData []float32, oh, ow int) ([]utils.Point, bool) {
	// Collect mask-positive points in resized coordinates.
	thr := r.cfg.MaskThreshold
	pts := make([]utils.Point, 0, (oh*ow)/8)
	mask := outData[2*oh*ow : 3*oh*ow]
	for y := range oh {
		row := y * ow
		for x := range ow {
			if float64(mask[row+x]) >= thr {
				pts = append(pts, utils.Point{X: float64(x), Y: float64(y)})
			}
		}
	}

	coverage := float64(len(pts)) / float64(oh*ow)
	if coverage < r.cfg.MinMaskCoverage || len(pts) < 100 {
		return nil, false
	}

	// Find minimum-area rectangle in resized space
	rect := utils.MinimumAreaRectangle(pts)
	if len(rect) != 4 {
		return nil, false
	}

	if r.cfg.DebugDir != "" {
		// Dump mask visualization (resized space)
		_ = dumpMaskPNG(r.cfg.DebugDir, mask, ow, oh, thr)
	}

	// Validate rectangle
	if !r.validateRectangle(rect, oh, ow) {
		return nil, false
	}

	return rect, true
}

// validateRectangle checks if the rectangle meets quality criteria.
func (r *Rectifier) validateRectangle(rect []utils.Point, oh, ow int) bool {
	// Gating based on rect area and aspect ratio in resized space
	rw0 := hypot(rect[1], rect[0])
	rw1 := hypot(rect[2], rect[3])
	rh0 := hypot(rect[3], rect[0])
	rh1 := hypot(rect[2], rect[1])
	ravgW := (rw0 + rw1) * 0.5
	ravgH := (rh0 + rh1) * 0.5

	if ravgW <= 1 || ravgH <= 1 {
		return false
	}

	rArea := ravgW * ravgH
	imgArea := float64(ow * oh)
	if rArea/imgArea < r.cfg.MinRectAreaRatio {
		return false
	}

	ar := ravgW / ravgH
	if ar < r.cfg.MinRectAspect || ar > r.cfg.MaxRectAspect {
		return false
	}

	return true
}

// hypot returns Euclidean distance between points a and b.
func hypot(a, b utils.Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

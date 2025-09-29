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

// processDocTROutput processes DocTR model output to extract document corners.
// DocTR typically outputs corner coordinates directly rather than a mask.
func (r *Rectifier) processDocTROutput(outData []float32, oh, ow int) ([]utils.Point, bool) {
	// DocTR models typically output 8 values representing 4 corner coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
	// The exact format may vary depending on the specific DocTR model implementation
	if len(outData) < 8 {
		return nil, false
	}

	// Extract corner coordinates from the first 8 values
	// Assuming output format: [x1, y1, x2, y2, x3, y3, x4, y4]
	corners := make([]utils.Point, 4)
	for i := range 4 {
		x := float64(outData[i*2])
		y := float64(outData[i*2+1])

		// Normalize coordinates if they are in [0,1] range
		if x >= 0 && x <= 1 && y >= 0 && y <= 1 {
			x *= float64(ow)
			y *= float64(oh)
		}

		// Clamp to image bounds
		if x < 0 {
			x = 0
		}
		if x >= float64(ow) {
			x = float64(ow - 1)
		}
		if y < 0 {
			y = 0
		}
		if y >= float64(oh) {
			y = float64(oh - 1)
		}

		corners[i] = utils.Point{X: x, Y: y}
	}

	// Validate that we have a reasonable quadrilateral
	if !r.validateDocTRCorners(corners, oh, ow) {
		return nil, false
	}

	return corners, true
}

// validateDocTRCorners validates that the predicted corners form a reasonable document quadrilateral.
func (r *Rectifier) validateDocTRCorners(corners []utils.Point, oh, ow int) bool {
	if len(corners) != 4 {
		return false
	}

	// Check that points are not too close together (degenerate quadrilateral)
	minDist := float64(ow) * 0.05 // Minimum 5% of image width between points
	for i := range 4 {
		for j := i + 1; j < 4; j++ {
			dist := hypot(corners[i], corners[j])
			if dist < minDist {
				return false
			}
		}
	}

	// Check aspect ratio (similar to UVDoc validation)
	// Calculate approximate width and height
	width := (hypot(corners[0], corners[1]) + hypot(corners[2], corners[3])) * 0.5
	height := (hypot(corners[0], corners[3]) + hypot(corners[1], corners[2])) * 0.5

	if width <= 1 || height <= 1 {
		return false
	}

	area := width * height
	imgArea := float64(ow * oh)
	if area/imgArea < r.cfg.MinRectAreaRatio {
		return false
	}

	ar := width / height
	if ar < r.cfg.MinRectAspect || ar > r.cfg.MaxRectAspect {
		return false
	}

	return true
}

// hypot returns Euclidean distance between points a and b.
func hypot(a, b utils.Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

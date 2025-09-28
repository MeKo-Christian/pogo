package detector

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

// DetectionResultJSON is a serializable representation of detected regions.
type DetectionResultJSON struct {
	Width   int          `json:"width"`
	Height  int          `json:"height"`
	Regions []RegionJSON `json:"regions"`
}

type RegionJSON struct {
	Confidence float64     `json:"confidence"`
	Box        BoxJSON     `json:"box"`
	Polygon    []PointJSON `json:"polygon,omitempty"`
}

type BoxJSON struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

type PointJSON struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// RegionsToJSON converts regions to JSON with the given image dimensions.
func RegionsToJSON(regs []DetectedRegion, width, height int) ([]byte, error) {
	out := DetectionResultJSON{Width: width, Height: height}
	out.Regions = make([]RegionJSON, 0, len(regs))
	for _, r := range regs {
		rr := RegionJSON{
			Confidence: r.Confidence,
			Box: BoxJSON{
				X: int(r.Box.MinX),
				Y: int(r.Box.MinY),
				W: int(r.Box.Width()),
				H: int(r.Box.Height()),
			},
		}
		for _, p := range r.Polygon {
			rr.Polygon = append(rr.Polygon, PointJSON{X: int(p.X), Y: int(p.Y)})
		}
		out.Regions = append(out.Regions, rr)
	}
	return json.MarshalIndent(out, "", "  ")
}

// RegionsFromJSON parses regions JSON into a struct.
func RegionsFromJSON(data []byte) (DetectionResultJSON, error) {
	var res DetectionResultJSON
	err := json.Unmarshal(data, &res)
	return res, err
}

// ValidateRegions performs basic sanity checks against image dimensions.
// validatePolygonPoints checks if all polygon points are within bounds.
func validatePolygonPoints(polygon []utils.Point, index, width, height int) error {
	for _, p := range polygon {
		if p.X < 0 || p.Y < 0 || p.X > float64(width) || p.Y > float64(height) {
			return fmt.Errorf("region %d polygon point out of bounds", index)
		}
	}
	return nil
}

// validateRegion checks if a single detected region is valid.
func validateRegion(r DetectedRegion, index, width, height int) error {
	if r.Box.Width() <= 0 || r.Box.Height() <= 0 {
		return fmt.Errorf("region %d has non-positive box size", index)
	}
	if r.Box.MinX < 0 || r.Box.MinY < 0 || r.Box.MaxX > float64(width) || r.Box.MaxY > float64(height) {
		return fmt.Errorf("region %d box out of bounds", index)
	}
	return validatePolygonPoints(r.Polygon, index, width, height)
}

func ValidateRegions(regs []DetectedRegion, width, height int) error {
	if width <= 0 || height <= 0 {
		return errors.New("invalid image dimensions for validation")
	}
	for i, r := range regs {
		if err := validateRegion(r, i, width, height); err != nil {
			return err
		}
	}
	return nil
}

// VisualizeOptions controls how regions are drawn onto images.
type VisualizeOptions struct {
	DrawBoxes   bool
	DrawPolygon bool
	Color       color.Color
	Thickness   int
}

// VisualizeRegions draws regions onto a copy of img and returns an RGBA image.
func VisualizeRegions(img image.Image, regs []DetectedRegion, opt VisualizeOptions) *image.RGBA {
	if opt.Color == nil {
		opt.Color = color.RGBA{255, 0, 0, 255}
	}
	if opt.Thickness <= 0 {
		opt.Thickness = 1
	}
	// Create a copy in RGBA
	b := img.Bounds()
	dst := image.NewRGBA(b)
	// Copy pixels
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x, y, img.At(x, y))
		}
	}
	for _, r := range regs {
		if opt.DrawBoxes {
			rect := r.Box.ToRect(b)
			utils.DrawRect(dst, rect, opt.Color, opt.Thickness)
		}
		if opt.DrawPolygon && len(r.Polygon) > 0 {
			utils.DrawPolygon(dst, r.Polygon, opt.Color, opt.Thickness)
		}
	}
	return dst
}

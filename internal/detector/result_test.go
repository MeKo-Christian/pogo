package detector

import (
	"image"
	"image/color"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
)

func makeDummyRegions() []DetectedRegion {
	return []DetectedRegion{
		{
			Box:        utils.NewBox(10, 10, 50, 40),
			Polygon:    []utils.Point{{X: 10, Y: 10}, {X: 50, Y: 10}, {X: 50, Y: 40}, {X: 10, Y: 40}},
			Confidence: 0.9,
		},
		{
			Box:        utils.NewBox(60, 5, 80, 20),
			Polygon:    []utils.Point{{X: 60, Y: 5}, {X: 80, Y: 5}, {X: 80, Y: 20}, {X: 60, Y: 20}},
			Confidence: 0.8,
		},
	}
}

func TestRegionsJSONRoundTrip(t *testing.T) {
	regs := makeDummyRegions()
	data, err := RegionsToJSON(regs, 100, 60)
	if err != nil {
		t.Fatalf("to JSON: %v", err)
	}
	parsed, err := RegionsFromJSON(data)
	if err != nil {
		t.Fatalf("from JSON: %v", err)
	}
	if parsed.Width != 100 || parsed.Height != 60 {
		t.Fatalf("dims mismatch: %dx%d", parsed.Width, parsed.Height)
	}
	if len(parsed.Regions) != len(regs) {
		t.Fatalf("region count mismatch: %d vs %d", len(parsed.Regions), len(regs))
	}
}

func TestValidateRegions(t *testing.T) {
	regs := makeDummyRegions()
	if err := ValidateRegions(regs, 100, 60); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	// Out of bounds
	regs[0].Box.MaxX = 1000
	if err := ValidateRegions(regs, 100, 60); err == nil {
		t.Fatalf("expected validation error for OOB")
	}
}

func TestVisualizeRegions(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 60))
	// fill white
	for y := range 60 {
		for x := range 100 {
			img.Set(x, y, color.RGBA{255, 255, 255, 255})
		}
	}
	regs := makeDummyRegions()
	out := VisualizeRegions(img, regs, VisualizeOptions{DrawBoxes: true, Color: color.RGBA{255, 0, 0, 255}, Thickness: 2})
	if out.Bounds().Dx() != 100 || out.Bounds().Dy() != 60 {
		t.Fatalf("unexpected size")
	}
	// Check a border pixel was colored
	c := out.RGBAAt(10, 10)
	if c.R == 255 && c.G == 255 && c.B == 255 {
		t.Fatalf("expected colored pixel on box edge")
	}
}

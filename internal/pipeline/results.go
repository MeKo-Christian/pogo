package pipeline

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ToJSONImage serializes a single OCRImageResult to pretty JSON.
func ToJSONImage(res *OCRImageResult) (string, error) {
	if res == nil {
		return "", errors.New("nil result")
	}
	b, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ToJSONImages serializes multiple OCRImageResult entries to pretty JSON.
func ToJSONImages(results []*OCRImageResult) (string, error) {
	b, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ToPlainTextImage extracts text lines from regions in detection order.
// Future improvement: implement reading order by y/x sorting and line grouping.
func ToPlainTextImage(res *OCRImageResult) (string, error) {
	if res == nil {
		return "", errors.New("nil result")
	}
	if len(res.Regions) == 0 {
		return "", nil
	}
	var lines []string
	lines = make([]string, 0, len(res.Regions))
	for _, r := range res.Regions {
		t := strings.TrimSpace(r.Text)
		if t != "" {
			lines = append(lines, t)
		}
	}
	return strings.Join(lines, "\n"), nil
}

// ToCSVImage exports per-region structured data as CSV with header.
func ToCSVImage(res *OCRImageResult) (string, error) {
	if res == nil {
		return "", errors.New("nil result")
	}
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"x", "y", "w", "h", "det_conf", "text", "rec_conf"})
	for _, r := range res.Regions {
		row := []string{
			strconv.Itoa(r.Box.X),
			strconv.Itoa(r.Box.Y),
			strconv.Itoa(r.Box.W),
			strconv.Itoa(r.Box.H),
			fmt.Sprintf("%.3f", r.DetConfidence),
			r.Text,
			fmt.Sprintf("%.3f", r.RecConfidence),
		}
		_ = w.Write(row)
	}
	w.Flush()
	return buf.String(), nil
}

// SortRegionsTopLeft sorts regions by top-left (y, then x) for readable ordering.
func SortRegionsTopLeft(res *OCRImageResult) {
	sort.SliceStable(res.Regions, func(i, j int) bool {
		if res.Regions[i].Box.Y == res.Regions[j].Box.Y {
			return res.Regions[i].Box.X < res.Regions[j].Box.X
		}
		return res.Regions[i].Box.Y < res.Regions[j].Box.Y
	})
}

// validateRegionBox checks if a region's bounding box is valid within the image bounds.
func validateRegionBox(r OCRRegionResult, imageWidth, imageHeight int, regionIndex int) error {
	if r.Box.W < 0 || r.Box.H < 0 {
		return fmt.Errorf("region %d has negative size", regionIndex)
	}
	if r.Box.X < 0 || r.Box.Y < 0 {
		return fmt.Errorf("region %d has negative coords", regionIndex)
	}
	if r.Box.X+r.Box.W > imageWidth {
		return fmt.Errorf("region %d exceeds image width", regionIndex)
	}
	if r.Box.Y+r.Box.H > imageHeight {
		return fmt.Errorf("region %d exceeds image height", regionIndex)
	}
	return nil
}

// validateRegionConfidence checks if a region's confidence values are in valid range.
func validateRegionConfidence(r OCRRegionResult, regionIndex int) error {
	if r.DetConfidence < 0 || r.DetConfidence > 1 {
		return fmt.Errorf("region %d det conf out of range", regionIndex)
	}
	if r.RecConfidence < 0 || r.RecConfidence > 1 {
		return fmt.Errorf("region %d rec conf out of range", regionIndex)
	}
	return nil
}

// ValidateOCRImageResult performs simple consistency checks.
func ValidateOCRImageResult(res *OCRImageResult) error {
	if res == nil {
		return errors.New("nil result")
	}
	if res.Width <= 0 || res.Height <= 0 {
		return fmt.Errorf("invalid image size %dx%d", res.Width, res.Height)
	}
	for i, r := range res.Regions {
		if err := validateRegionBox(r, res.Width, res.Height, i); err != nil {
			return err
		}
		if err := validateRegionConfidence(r, i); err != nil {
			return err
		}
	}
	return nil
}

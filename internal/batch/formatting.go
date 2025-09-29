package batch

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// formatBatchResults formats the batch processing results in the specified format.
func formatBatchResults(results []*pipeline.OCRImageResult, imagePaths []string, format string) (string, error) {
	switch format {
	case "json":
		return formatJSON(results, imagePaths)
	case "csv":
		return formatCSV(results, imagePaths)
	default: // text
		return formatText(results, imagePaths)
	}
}

// formatJSON formats results as JSON.
func formatJSON(results []*pipeline.OCRImageResult, imagePaths []string) (string, error) {
	batchResult := struct {
		Images []struct {
			File string                   `json:"file"`
			OCR  *pipeline.OCRImageResult `json:"ocr"`
		} `json:"images"`
	}{}

	batchResult.Images = make([]struct {
		File string                   `json:"file"`
		OCR  *pipeline.OCRImageResult `json:"ocr"`
	}, len(results))

	for i, res := range results {
		batchResult.Images[i] = struct {
			File string                   `json:"file"`
			OCR  *pipeline.OCRImageResult `json:"ocr"`
		}{
			File: imagePaths[i],
			OCR:  res,
		}
	}

	bts, err := json.MarshalIndent(batchResult, "", "  ")
	return string(bts), err
}

// formatCSV formats results as CSV.
func formatCSV(results []*pipeline.OCRImageResult, imagePaths []string) (string, error) {
	var csvData [][]string
	// Header
	csvData = append(csvData, []string{
		"file", "region_index", "text", "confidence", "det_confidence", "x", "y", "width", "height", "language",
	})

	for i, res := range results {
		if res == nil {
			continue
		}
		file := imagePaths[i]
		if len(res.Regions) == 0 {
			// Add empty row for files with no regions
			csvData = append(csvData, []string{file, "0", "", "0", "0", "0", "0", "0", "0", ""})
		} else {
			for j, region := range res.Regions {
				csvData = append(csvData, []string{
					file,
					strconv.Itoa(j),
					region.Text,
					fmt.Sprintf("%.3f", region.RecConfidence),
					fmt.Sprintf("%.3f", region.DetConfidence),
					strconv.Itoa(region.Box.X),
					strconv.Itoa(region.Box.Y),
					strconv.Itoa(region.Box.W),
					strconv.Itoa(region.Box.H),
					region.Language,
				})
			}
		}
	}

	var output strings.Builder
	writer := csv.NewWriter(&output)
	for _, row := range csvData {
		if err := writer.Write(row); err != nil {
			return "", err
		}
	}
	writer.Flush()
	return output.String(), nil
}

// formatText formats results as plain text.
func formatText(results []*pipeline.OCRImageResult, imagePaths []string) (string, error) {
	var output strings.Builder
	for i, res := range results {
		if i > 0 {
			output.WriteString("\n")
		}
		output.WriteString(fmt.Sprintf("# %s\n", imagePaths[i]))
		if res == nil {
			continue
		}
		pipeline.SortRegionsTopLeft(res)
		text, err := pipeline.ToPlainTextImage(res)
		if err != nil {
			return "", err
		}
		output.WriteString(text)
	}
	return output.String(), nil
}

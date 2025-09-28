package pdf

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPageResult_Serialization(t *testing.T) {
	regions := []detector.DetectedRegion{
		{
			Box:        utils.Box{MinX: 10, MinY: 20, MaxX: 100, MaxY: 60},
			Confidence: 0.95,
			Polygon: []utils.Point{
				{X: 10, Y: 20},
				{X: 100, Y: 20},
				{X: 100, Y: 60},
				{X: 10, Y: 60},
			},
		},
		{
			Box:        utils.Box{MinX: 50, MinY: 80, MaxX: 150, MaxY: 110},
			Confidence: 0.87,
		},
	}

	pageResult := PageResult{
		PageNumber: 1,
		Width:      200,
		Height:     300,
		Images: []ImageResult{
			{
				ImageIndex: 0,
				Width:      200,
				Height:     300,
				Regions:    regions,
				Confidence: 0.91,
			},
		},
		Processing: ProcessingInfo{
			ExtractionTimeMs: 150,
			DetectionTimeMs:  250,
			TotalTimeMs:      400,
		},
	}

	t.Run("marshal to JSON", func(t *testing.T) {
		data, err := json.Marshal(pageResult)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"page_number":1`)
		assert.Contains(t, string(data), `"width":200`)
		assert.Contains(t, string(data), `"height":300`)
		assert.Contains(t, string(data), `"confidence":0.91`)
	})

	t.Run("unmarshal from JSON", func(t *testing.T) {
		data, err := json.Marshal(pageResult)
		require.NoError(t, err)

		var unmarshaled PageResult
		err = json.Unmarshal(data, &unmarshaled)
		require.NoError(t, err)

		assert.Equal(t, pageResult.PageNumber, unmarshaled.PageNumber)
		assert.Equal(t, pageResult.Width, unmarshaled.Width)
		assert.Equal(t, pageResult.Height, unmarshaled.Height)
		assert.Len(t, unmarshaled.Images, 1)
		assert.InDelta(t, pageResult.Images[0].Confidence, unmarshaled.Images[0].Confidence, 0.0001)
	})
}

func TestImageResult_Serialization(t *testing.T) {
	regions := []detector.DetectedRegion{
		{
			Box:        utils.Box{MinX: 5, MinY: 10, MaxX: 95, MaxY: 40},
			Confidence: 0.88,
		},
	}

	imageResult := ImageResult{
		ImageIndex: 1,
		Width:      150,
		Height:     200,
		Regions:    regions,
		Confidence: 0.88,
	}

	t.Run("marshal to JSON", func(t *testing.T) {
		data, err := json.Marshal(imageResult)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"image_index":1`)
		assert.Contains(t, string(data), `"width":150`)
		assert.Contains(t, string(data), `"height":200`)
		assert.Contains(t, string(data), `"confidence":0.88`)
	})

	t.Run("unmarshal from JSON", func(t *testing.T) {
		data, err := json.Marshal(imageResult)
		require.NoError(t, err)

		var unmarshaled ImageResult
		err = json.Unmarshal(data, &unmarshaled)
		require.NoError(t, err)

		assert.Equal(t, imageResult.ImageIndex, unmarshaled.ImageIndex)
		assert.Equal(t, imageResult.Width, unmarshaled.Width)
		assert.Equal(t, imageResult.Height, unmarshaled.Height)
		assert.InDelta(t, imageResult.Confidence, unmarshaled.Confidence, 0.0001)
		assert.Len(t, unmarshaled.Regions, 1)
	})
}

func TestDocumentResult_Serialization(t *testing.T) {
	documentResult := DocumentResult{
		Filename:   "test.pdf",
		TotalPages: 2,
		Pages: []PageResult{
			{
				PageNumber: 1,
				Width:      200,
				Height:     300,
				Images: []ImageResult{
					{
						ImageIndex: 0,
						Width:      200,
						Height:     300,
						Regions:    []detector.DetectedRegion{},
						Confidence: 0.85,
					},
				},
				Processing: ProcessingInfo{
					DetectionTimeMs: 100,
					TotalTimeMs:     100,
				},
			},
			{
				PageNumber: 2,
				Width:      200,
				Height:     300,
				Images:     []ImageResult{},
				Processing: ProcessingInfo{
					DetectionTimeMs: 50,
					TotalTimeMs:     50,
				},
			},
		},
		Processing: ProcessingInfo{
			ExtractionTimeMs: 200,
			DetectionTimeMs:  150,
			TotalTimeMs:      350,
		},
	}

	t.Run("marshal to JSON", func(t *testing.T) {
		data, err := json.Marshal(documentResult)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"filename":"test.pdf"`)
		assert.Contains(t, string(data), `"total_pages":2`)
		assert.Contains(t, string(data), `"extraction_time_ms":200`)
	})

	t.Run("unmarshal from JSON", func(t *testing.T) {
		data, err := json.Marshal(documentResult)
		require.NoError(t, err)

		var unmarshaled DocumentResult
		err = json.Unmarshal(data, &unmarshaled)
		require.NoError(t, err)

		assert.Equal(t, documentResult.Filename, unmarshaled.Filename)
		assert.Equal(t, documentResult.TotalPages, unmarshaled.TotalPages)
		assert.Len(t, unmarshaled.Pages, 2)
		assert.Equal(t, documentResult.Processing.ExtractionTimeMs, unmarshaled.Processing.ExtractionTimeMs)
	})
}

func TestProcessingInfo_Serialization(t *testing.T) {
	processingInfo := ProcessingInfo{
		ExtractionTimeMs: 100,
		DetectionTimeMs:  250,
		TotalTimeMs:      350,
	}

	t.Run("marshal to JSON", func(t *testing.T) {
		data, err := json.Marshal(processingInfo)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"extraction_time_ms":100`)
		assert.Contains(t, string(data), `"detection_time_ms":250`)
		assert.Contains(t, string(data), `"total_time_ms":350`)
	})

	t.Run("unmarshal from JSON", func(t *testing.T) {
		data, err := json.Marshal(processingInfo)
		require.NoError(t, err)

		var unmarshaled ProcessingInfo
		err = json.Unmarshal(data, &unmarshaled)
		require.NoError(t, err)

		assert.Equal(t, processingInfo.ExtractionTimeMs, unmarshaled.ExtractionTimeMs)
		assert.Equal(t, processingInfo.DetectionTimeMs, unmarshaled.DetectionTimeMs)
		assert.Equal(t, processingInfo.TotalTimeMs, unmarshaled.TotalTimeMs)
	})
}

func TestPageResult_EmptyResults(t *testing.T) {
	pageResult := PageResult{
		PageNumber: 1,
		Width:      0,
		Height:     0,
		Images:     []ImageResult{},
		Processing: ProcessingInfo{},
	}

	data, err := json.Marshal(pageResult)
	require.NoError(t, err)

	var unmarshaled PageResult
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, pageResult.PageNumber, unmarshaled.PageNumber)
	assert.Equal(t, 0, unmarshaled.Width)
	assert.Equal(t, 0, unmarshaled.Height)
	assert.Empty(t, unmarshaled.Images)
}

func TestImageResult_EmptyRegions(t *testing.T) {
	imageResult := ImageResult{
		ImageIndex: 0,
		Width:      100,
		Height:     100,
		Regions:    []detector.DetectedRegion{},
		Confidence: 0.0,
	}

	data, err := json.Marshal(imageResult)
	require.NoError(t, err)

	var unmarshaled ImageResult
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, imageResult.ImageIndex, unmarshaled.ImageIndex)
	assert.Empty(t, unmarshaled.Regions)
	assert.InDelta(t, 0.0, unmarshaled.Confidence, 0.0001)
}

func TestDocumentResult_EmptyPages(t *testing.T) {
	documentResult := DocumentResult{
		Filename:   "empty.pdf",
		TotalPages: 0,
		Pages:      []PageResult{},
		Processing: ProcessingInfo{
			TotalTimeMs: 10,
		},
	}

	data, err := json.Marshal(documentResult)
	require.NoError(t, err)

	var unmarshaled DocumentResult
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, "empty.pdf", unmarshaled.Filename)
	assert.Equal(t, 0, unmarshaled.TotalPages)
	assert.Empty(t, unmarshaled.Pages)
}

func TestProcessingInfo_ZeroTimes(t *testing.T) {
	processingInfo := ProcessingInfo{
		ExtractionTimeMs: 0,
		DetectionTimeMs:  0,
		TotalTimeMs:      0,
	}

	data, err := json.Marshal(processingInfo)
	require.NoError(t, err)

	var unmarshaled ProcessingInfo
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, int64(0), unmarshaled.ExtractionTimeMs)
	assert.Equal(t, int64(0), unmarshaled.DetectionTimeMs)
	assert.Equal(t, int64(0), unmarshaled.TotalTimeMs)
}

func TestProcessingInfo_LargeTimes(t *testing.T) {
	processingInfo := ProcessingInfo{
		ExtractionTimeMs: time.Hour.Milliseconds(),
		DetectionTimeMs:  time.Minute.Milliseconds() * 30,
		TotalTimeMs:      time.Hour.Milliseconds() + time.Minute.Milliseconds()*30,
	}

	data, err := json.Marshal(processingInfo)
	require.NoError(t, err)

	var unmarshaled ProcessingInfo
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, processingInfo.ExtractionTimeMs, unmarshaled.ExtractionTimeMs)
	assert.Equal(t, processingInfo.DetectionTimeMs, unmarshaled.DetectionTimeMs)
	assert.Equal(t, processingInfo.TotalTimeMs, unmarshaled.TotalTimeMs)
}

// Test edge cases with JSON field names.
func TestJSON_FieldNames(t *testing.T) {
	pageResult := PageResult{
		PageNumber: 42,
		Width:      640,
		Height:     480,
	}

	data, err := json.Marshal(pageResult)
	require.NoError(t, err)

	// Verify that JSON uses snake_case field names
	jsonStr := string(data)
	assert.Contains(t, jsonStr, `"page_number":42`)
	assert.Contains(t, jsonStr, `"width":640`)
	assert.Contains(t, jsonStr, `"height":480`)

	// Should not contain camelCase
	assert.NotContains(t, jsonStr, `"pageNumber"`)
	assert.NotContains(t, jsonStr, `"PageNumber"`)
}

func TestImageResult_WithComplexRegions(t *testing.T) {
	// Test with regions that have complex polygons
	regions := []detector.DetectedRegion{
		{
			Box: utils.Box{MinX: 10, MinY: 20, MaxX: 100, MaxY: 60},
			Polygon: []utils.Point{
				{X: 10, Y: 20},
				{X: 50, Y: 15}, // irregular polygon
				{X: 100, Y: 25},
				{X: 95, Y: 60},
				{X: 15, Y: 55},
			},
			Confidence: 0.92,
		},
	}

	imageResult := ImageResult{
		ImageIndex: 0,
		Width:      200,
		Height:     300,
		Regions:    regions,
		Confidence: 0.92,
	}

	// Test that complex regions serialize/deserialize correctly
	data, err := json.Marshal(imageResult)
	require.NoError(t, err)

	var unmarshaled ImageResult
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Len(t, unmarshaled.Regions, 1)
	assert.InDelta(t, regions[0].Confidence, unmarshaled.Regions[0].Confidence, 0.0001)
	// Note: We can't easily test polygon equality without comparing each point
	// but this verifies the structure serializes without error
}

// Benchmark tests.
func BenchmarkPageResult_Marshal(b *testing.B) {
	pageResult := PageResult{
		PageNumber: 1,
		Width:      800,
		Height:     600,
		Images: []ImageResult{
			{
				ImageIndex: 0,
				Width:      800,
				Height:     600,
				Regions:    make([]detector.DetectedRegion, 100), // many regions
				Confidence: 0.85,
			},
		},
		Processing: ProcessingInfo{
			ExtractionTimeMs: 150,
			DetectionTimeMs:  250,
			TotalTimeMs:      400,
		},
	}

	b.ResetTimer()
	for range b.N {
		if _, err := json.Marshal(pageResult); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDocumentResult_Marshal(b *testing.B) {
	// Create a document with multiple pages for benchmarking
	pages := make([]PageResult, 50)
	for i := range pages {
		pages[i] = PageResult{
			PageNumber: i + 1,
			Width:      800,
			Height:     600,
			Images: []ImageResult{
				{
					ImageIndex: 0,
					Width:      800,
					Height:     600,
					Regions:    make([]detector.DetectedRegion, 10),
					Confidence: 0.85,
				},
			},
		}
	}

	documentResult := DocumentResult{
		Filename:   "large_document.pdf",
		TotalPages: len(pages),
		Pages:      pages,
		Processing: ProcessingInfo{
			ExtractionTimeMs: 1000,
			DetectionTimeMs:  5000,
			TotalTimeMs:      6000,
		},
	}

	b.ResetTimer()
	for range b.N {
		if _, err := json.Marshal(documentResult); err != nil {
			b.Fatal(err)
		}
	}
}

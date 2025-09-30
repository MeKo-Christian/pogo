package pdf

import (
	"strings"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMergeStrategy_String(t *testing.T) {
	tests := []struct {
		name     string
		strategy MergeStrategy
		want     string
	}{
		{
			name:     "append strategy",
			strategy: MergeStrategyAppend,
			want:     "append",
		},
		{
			name:     "spatial strategy",
			strategy: MergeStrategySpatial,
			want:     "spatial",
		},
		{
			name:     "confidence strategy",
			strategy: MergeStrategyConfidence,
			want:     "confidence",
		},
		{
			name:     "hybrid smart strategy",
			strategy: MergeStrategyHybridSmart,
			want:     "hybrid_smart",
		},
		{
			name:     "unknown strategy",
			strategy: MergeStrategy(999),
			want:     "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.strategy.String())
		})
	}
}

func TestDefaultHybridConfig(t *testing.T) {
	config := DefaultHybridConfig()
	require.NotNil(t, config)

	assert.Equal(t, MergeStrategyHybridSmart, config.MergeStrategy)
	assert.InDelta(t, 0.5, config.ConfidenceThreshold, 1e-9)
	assert.InDelta(t, 0.1, config.SpatialOverlapThreshold, 1e-9)
	assert.True(t, config.DeduplicationEnabled)
	assert.InDelta(t, 0.8, config.DeduplicationSimilarity, 1e-9)
	assert.True(t, config.PreferVectorText)
	assert.InDelta(t, 0.3, config.MinOCRConfidence, 1e-9)
}

func TestNewHybridProcessor(t *testing.T) {
	t.Run("create with nil config", func(t *testing.T) {
		processor := NewHybridProcessor(nil)
		require.NotNil(t, processor)
		require.NotNil(t, processor.config)

		// Should use default config
		assert.Equal(t, MergeStrategyHybridSmart, processor.config.MergeStrategy)
	})

	t.Run("create with custom config", func(t *testing.T) {
		customConfig := &HybridConfig{
			MergeStrategy:           MergeStrategyAppend,
			ConfidenceThreshold:     0.7,
			SpatialOverlapThreshold: 0.2,
			DeduplicationEnabled:    false,
			DeduplicationSimilarity: 0.9,
			PreferVectorText:        false,
			MinOCRConfidence:        0.4,
		}

		processor := NewHybridProcessor(customConfig)
		require.NotNil(t, processor)
		assert.Equal(t, customConfig, processor.config)
		assert.Equal(t, MergeStrategyAppend, processor.config.MergeStrategy)
		assert.InDelta(t, 0.7, processor.config.ConfidenceThreshold, 1e-9)
	})
}

func TestHybridProcessor_GetConfig(t *testing.T) {
	processor := NewHybridProcessor(nil)
	config := processor.GetConfig()

	require.NotNil(t, config)
	assert.Equal(t, processor.config, config)
}

func TestHybridProcessor_UpdateConfig(t *testing.T) {
	processor := NewHybridProcessor(nil)
	originalConfig := processor.config

	t.Run("update with new config", func(t *testing.T) {
		newConfig := &HybridConfig{
			MergeStrategy:       MergeStrategyConfidence,
			ConfidenceThreshold: 0.9,
		}

		processor.UpdateConfig(newConfig)
		assert.Equal(t, newConfig, processor.config)
		assert.NotEqual(t, originalConfig, processor.config)
	})

	t.Run("update with nil config", func(t *testing.T) {
		currentConfig := processor.config
		processor.UpdateConfig(nil)
		// Config should remain unchanged
		assert.Equal(t, currentConfig, processor.config)
	})
}

func TestHybridProcessor_MergeResults_ErrorCases(t *testing.T) {
	processor := NewHybridProcessor(nil)

	t.Run("no input data", func(t *testing.T) {
		result, err := processor.MergeResults(nil, nil, 800, 600)
		require.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "no input data provided")
	})

	t.Run("empty OCR results slice", func(t *testing.T) {
		result, err := processor.MergeResults(nil, []ImageResult{}, 800, 600)
		require.Error(t, err)
		assert.Nil(t, result)
	})
}

func TestHybridProcessor_DetermineStrategy(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name       string
		vectorText *TextExtraction
		ocrResults []ImageResult
		want       ProcessingStrategy
	}{
		{
			name: "both sources available",
			vectorText: &TextExtraction{
				Text: "Some text",
				Quality: TextQuality{
					HasText: true,
				},
			},
			ocrResults: []ImageResult{{ImageIndex: 0}},
			want:       StrategyHybrid,
		},
		{
			name: "vector text only",
			vectorText: &TextExtraction{
				Text: "Some text",
				Quality: TextQuality{
					HasText: true,
				},
			},
			ocrResults: nil,
			want:       StrategyVectorText,
		},
		{
			name:       "OCR only",
			vectorText: nil,
			ocrResults: []ImageResult{{ImageIndex: 0}},
			want:       StrategyOCR,
		},
		{
			name:       "neither source",
			vectorText: nil,
			ocrResults: nil,
			want:       StrategySkip,
		},
		{
			name: "vector text without actual text",
			vectorText: &TextExtraction{
				Text: "",
				Quality: TextQuality{
					HasText: false,
				},
			},
			ocrResults: []ImageResult{{ImageIndex: 0}},
			want:       StrategyOCR,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := processor.determineStrategy(tt.vectorText, tt.ocrResults)
			assert.Equal(t, tt.want, strategy)
		})
	}
}

func TestHybridProcessor_MergeByAppending(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy:    MergeStrategyAppend,
		MinOCRConfidence: 0.3,
	})

	t.Run("vector text only", func(t *testing.T) {
		result, err := processor.MergeResults(
			&TextExtraction{
				PageNumber: 1,
				Text:       "Vector text content",
				Quality:    TextQuality{HasText: true, Score: 0.9},
			},
			nil,
			800, 600,
		)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, "Vector text content", result.CombinedText)
		assert.Equal(t, 1, result.PageNumber)
		assert.True(t, result.ProcessingInfo.VectorTextUsed)
		assert.False(t, result.ProcessingInfo.OCRUsed)
	})

	t.Run("OCR only with OCRRegions", func(t *testing.T) {
		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "OCR text 1",
						RecConfidence: 0.85,
					},
					{
						Text:          "OCR text 2",
						RecConfidence: 0.75,
					},
				},
			},
		}

		result, err := processor.MergeResults(nil, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, "OCR text 1")
		assert.Contains(t, result.CombinedText, "OCR text 2")
		assert.False(t, result.ProcessingInfo.VectorTextUsed)
		assert.True(t, result.ProcessingInfo.OCRUsed)
		assert.Len(t, result.MergedRegions, 2)
	})

	t.Run("both vector and OCR", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "Vector text",
			Quality:    TextQuality{HasText: true, Score: 0.9},
		}

		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "OCR text",
						RecConfidence: 0.8,
					},
				},
			},
		}

		result, err := processor.MergeResults(vectorText, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, "Vector text")
		assert.Contains(t, result.CombinedText, "OCR text")
		assert.True(t, result.ProcessingInfo.VectorTextUsed)
		assert.True(t, result.ProcessingInfo.OCRUsed)
	})

	t.Run("filter low confidence OCR", func(t *testing.T) {
		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "High confidence",
						RecConfidence: 0.9,
					},
					{
						Text:          "Low confidence",
						RecConfidence: 0.1,
					},
				},
			},
		}

		result, err := processor.MergeResults(nil, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, "High confidence")
		assert.NotContains(t, result.CombinedText, "Low confidence")
		assert.Len(t, result.MergedRegions, 1)
	})

	t.Run("OCR with DetectedRegions fallback", func(t *testing.T) {
		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				Regions: []detector.DetectedRegion{
					{
						Box: utils.Box{
							MinX: 10, MinY: 20, MaxX: 100, MaxY: 50,
						},
						Confidence: 0.85,
					},
				},
			},
		}

		result, err := processor.MergeResults(nil, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Len(t, result.MergedRegions, 1)
		assert.Equal(t, 10, result.MergedRegions[0].Box.X)
		assert.Equal(t, 20, result.MergedRegions[0].Box.Y)
		assert.Equal(t, 90, result.MergedRegions[0].Box.W)
		assert.Equal(t, 30, result.MergedRegions[0].Box.H)
	})
}

func TestHybridProcessor_MergeBySpatialLayout(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy:           MergeStrategySpatial,
		SpatialOverlapThreshold: 0.1,
		PreferVectorText:        true,
		MinOCRConfidence:        0.3,
	})

	t.Run("merge with spatial positioning", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "Vector text",
			Quality:    TextQuality{HasText: true, Score: 0.9},
			Positions: []TextPosition{
				{Text: "Vector text", X: 100, Y: 100, Width: 200, Height: 50},
			},
		}

		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "OCR text",
						Box:           struct{ X, Y, W, H int }{X: 300, Y: 100, W: 150, H: 40},
						RecConfidence: 0.8,
					},
				},
			},
		}

		result, err := processor.MergeResults(vectorText, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEmpty(t, result.CombinedText)
		assert.True(t, result.ProcessingInfo.SpatialMergingUsed)
	})

	t.Run("overlapping elements prefer vector", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "Vector",
			Quality:    TextQuality{HasText: true, Score: 0.9},
			Positions: []TextPosition{
				{Text: "Vector", X: 100, Y: 100, Width: 100, Height: 50},
			},
		}

		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "OCR",
						Box:           struct{ X, Y, W, H int }{X: 90, Y: 95, W: 100, H: 50},
						RecConfidence: 0.8,
					},
				},
			},
		}

		result, err := processor.MergeResults(vectorText, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		// Should prefer vector text due to overlap and PreferVectorText=true
		assert.NotEmpty(t, result.CombinedText)
	})
}

func TestHybridProcessor_MergeByConfidence(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy:       MergeStrategyConfidence,
		ConfidenceThreshold: 0.7,
	})

	t.Run("filter by confidence threshold", func(t *testing.T) {
		ocrResults := []ImageResult{
			{
				ImageIndex: 0,
				OCRRegions: []OCRRegion{
					{
						Text:          "High confidence text",
						RecConfidence: 0.9,
					},
					{
						Text:          "Medium confidence text",
						RecConfidence: 0.75,
					},
					{
						Text:          "Low confidence text",
						RecConfidence: 0.5,
					},
				},
			},
		}

		result, err := processor.MergeResults(nil, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, "High confidence text")
		assert.Contains(t, result.CombinedText, "Medium confidence text")
		assert.NotContains(t, result.CombinedText, "Low confidence text")
	})
}

func TestHybridProcessor_TextSimilarity(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name      string
		a         string
		b         string
		wantRange [2]float64 // min and max expected values
	}{
		{
			name:      "identical strings",
			a:         "hello world",
			b:         "hello world",
			wantRange: [2]float64{0.99, 1.0},
		},
		{
			name:      "completely different",
			a:         "hello world",
			b:         "foo bar",
			wantRange: [2]float64{0.0, 0.1},
		},
		{
			name:      "partial overlap",
			a:         "hello world test",
			b:         "hello world example",
			wantRange: [2]float64{0.4, 0.6},
		},
		{
			name:      "case insensitive",
			a:         "HELLO WORLD",
			b:         "hello world",
			wantRange: [2]float64{0.99, 1.0},
		},
		{
			name:      "both empty",
			a:         "",
			b:         "",
			wantRange: [2]float64{1.0, 1.0},
		},
		{
			name:      "one empty",
			a:         "hello",
			b:         "",
			wantRange: [2]float64{0.0, 0.0},
		},
		{
			name:      "different word order",
			a:         "world hello",
			b:         "hello world",
			wantRange: [2]float64{0.99, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			similarity := processor.textSimilarity(tt.a, tt.b)
			assert.GreaterOrEqual(t, similarity, tt.wantRange[0])
			assert.LessOrEqual(t, similarity, tt.wantRange[1])
		})
	}
}

func TestHybridProcessor_DeduplicateText(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		DeduplicationEnabled:    true,
		DeduplicationSimilarity: 0.8,
	})

	t.Run("remove exact duplicates", func(t *testing.T) {
		result := &HybridResult{
			CombinedText: "Line 1\nLine 1\nLine 2",
		}

		processor.deduplicateText(result)

		lines := strings.Split(result.CombinedText, "\n")
		assert.Len(t, lines, 2)
		assert.Contains(t, result.CombinedText, "Line 1")
		assert.Contains(t, result.CombinedText, "Line 2")
	})

	t.Run("remove similar lines", func(t *testing.T) {
		result := &HybridResult{
			CombinedText: "Hello world test\nHello world test example\nCompletely different",
		}

		processor.deduplicateText(result)

		// Should keep lines that are sufficiently different
		assert.Contains(t, result.CombinedText, "Completely different")
	})

	t.Run("empty text", func(t *testing.T) {
		result := &HybridResult{
			CombinedText: "",
		}

		processor.deduplicateText(result)
		assert.Empty(t, result.CombinedText)
	})

	t.Run("remove empty lines", func(t *testing.T) {
		result := &HybridResult{
			CombinedText: "Line 1\n\n\nLine 2\n",
		}

		processor.deduplicateText(result)

		lines := strings.Split(result.CombinedText, "\n")
		for _, line := range lines {
			assert.NotEmpty(t, strings.TrimSpace(line))
		}
	})
}

func TestHybridProcessor_ElementsOverlap(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		SpatialOverlapThreshold: 0.1,
	})

	tests := []struct {
		name    string
		a       TextElement
		b       TextElement
		wantOvl bool
	}{
		{
			name:    "no overlap",
			a:       TextElement{X: 100, Y: 100, Width: 50, Height: 50},
			b:       TextElement{X: 200, Y: 200, Width: 50, Height: 50},
			wantOvl: false,
		},
		{
			name:    "complete overlap",
			a:       TextElement{X: 100, Y: 100, Width: 50, Height: 50},
			b:       TextElement{X: 100, Y: 100, Width: 50, Height: 50},
			wantOvl: true,
		},
		{
			name:    "partial overlap above threshold",
			a:       TextElement{X: 100, Y: 100, Width: 100, Height: 100},
			b:       TextElement{X: 120, Y: 120, Width: 100, Height: 100},
			wantOvl: true,
		},
		{
			name:    "adjacent but not overlapping",
			a:       TextElement{X: 100, Y: 100, Width: 50, Height: 50},
			b:       TextElement{X: 150, Y: 100, Width: 50, Height: 50},
			wantOvl: false,
		},
		{
			name:    "small element inside large element",
			a:       TextElement{X: 100, Y: 100, Width: 200, Height: 200},
			b:       TextElement{X: 150, Y: 150, Width: 20, Height: 20},
			wantOvl: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overlap := processor.elementsOverlap(tt.a, tt.b)
			assert.Equal(t, tt.wantOvl, overlap)
		})
	}
}

func TestHybridProcessor_TextAlreadyIncluded(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name         string
		text         string
		combinedText string
		want         bool
	}{
		{
			name:         "text is included",
			text:         "hello",
			combinedText: "This is hello world",
			want:         true,
		},
		{
			name:         "text not included",
			text:         "foo",
			combinedText: "This is hello world",
			want:         false,
		},
		{
			name:         "case insensitive",
			text:         "HELLO",
			combinedText: "This is hello world",
			want:         true,
		},
		{
			name:         "empty text",
			text:         "",
			combinedText: "Some content",
			want:         true, // Empty string is always "contained"
		},
		{
			name:         "text with whitespace",
			text:         "  hello  ",
			combinedText: "This is hello world",
			want:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := processor.textAlreadyIncluded(tt.text, tt.combinedText)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestHybridProcessor_RegionAlreadyIncluded(t *testing.T) {
	processor := NewHybridProcessor(nil)

	region1 := OCRRegion{
		Text: "Test",
		Box:  struct{ X, Y, W, H int }{X: 10, Y: 20, W: 100, H: 50},
	}

	region2 := OCRRegion{
		Text: "Test",
		Box:  struct{ X, Y, W, H int }{X: 10, Y: 20, W: 100, H: 50},
	}

	region3 := OCRRegion{
		Text: "Different",
		Box:  struct{ X, Y, W, H int }{X: 10, Y: 20, W: 100, H: 50},
	}

	region4 := OCRRegion{
		Text: "Test",
		Box:  struct{ X, Y, W, H int }{X: 30, Y: 40, W: 100, H: 50},
	}

	t.Run("region is included", func(t *testing.T) {
		regions := []OCRRegion{region1}
		result := processor.regionAlreadyIncluded(region2, regions)
		assert.True(t, result)
	})

	t.Run("region not included - different text", func(t *testing.T) {
		regions := []OCRRegion{region1}
		result := processor.regionAlreadyIncluded(region3, regions)
		assert.False(t, result)
	})

	t.Run("region not included - different position", func(t *testing.T) {
		regions := []OCRRegion{region1}
		result := processor.regionAlreadyIncluded(region4, regions)
		assert.False(t, result)
	})

	t.Run("empty regions list", func(t *testing.T) {
		regions := []OCRRegion{}
		result := processor.regionAlreadyIncluded(region1, regions)
		assert.False(t, result)
	})
}

func TestHybridProcessor_CalculateQualityMetrics(t *testing.T) {
	processor := NewHybridProcessor(nil)

	t.Run("metrics with vector text and OCR", func(t *testing.T) {
		result := &HybridResult{
			VectorText: &TextExtraction{
				Text: "Vector text content",
				Quality: TextQuality{
					Score: 0.9,
				},
			},
			OCRResults: []ImageResult{
				{
					OCRRegions: []OCRRegion{
						{Text: "OCR text", RecConfidence: 0.8},
					},
				},
			},
			MergedRegions: []OCRRegion{
				{Text: "OCR text", RecConfidence: 0.8},
			},
			CombinedText: "Vector text content OCR text",
		}

		processor.calculateQualityMetrics(result)

		assert.Greater(t, result.QualityMetrics.OverallScore, 0.0)
		assert.LessOrEqual(t, result.QualityMetrics.OverallScore, 1.0)
		assert.Greater(t, result.QualityMetrics.VectorTextContrib, 0.0)
		assert.Greater(t, result.QualityMetrics.OCRContrib, 0.0)
		assert.Greater(t, result.QualityMetrics.TextCoverage, 0.0)
		assert.Greater(t, result.QualityMetrics.ConfidenceAverage, 0.0)
	})

	t.Run("metrics with vector text only", func(t *testing.T) {
		result := &HybridResult{
			VectorText: &TextExtraction{
				Text: "Vector text only",
				Quality: TextQuality{
					Score: 0.95,
				},
			},
			CombinedText: "Vector text only",
		}

		processor.calculateQualityMetrics(result)

		assert.Greater(t, result.QualityMetrics.OverallScore, 0.0)
		assert.InDelta(t, 1.0, result.QualityMetrics.VectorTextContrib, 1e-9)
		assert.InDelta(t, 0.0, result.QualityMetrics.OCRContrib, 1e-9)
	})

	t.Run("metrics with OCR only", func(t *testing.T) {
		result := &HybridResult{
			OCRResults: []ImageResult{
				{
					OCRRegions: []OCRRegion{
						{Text: "OCR only", RecConfidence: 0.85},
					},
				},
			},
			MergedRegions: []OCRRegion{
				{Text: "OCR only", RecConfidence: 0.85},
			},
			CombinedText: "OCR only",
		}

		processor.calculateQualityMetrics(result)

		assert.Greater(t, result.QualityMetrics.OverallScore, 0.0)
		assert.InDelta(t, 0.0, result.QualityMetrics.VectorTextContrib, 1e-9)
		assert.InDelta(t, 1.0, result.QualityMetrics.OCRContrib, 1e-9)
	})
}

func TestHybridProcessor_CalculateOverallScore(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name   string
		result *HybridResult
		min    float64
		max    float64
	}{
		{
			name: "high quality vector and OCR",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Quality: TextQuality{Score: 0.9},
				},
				MergedRegions: []OCRRegion{
					{RecConfidence: 0.9},
					{RecConfidence: 0.85},
				},
			},
			min: 0.8,
			max: 1.0,
		},
		{
			name: "low quality sources",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Quality: TextQuality{Score: 0.3},
				},
				MergedRegions: []OCRRegion{
					{RecConfidence: 0.4},
				},
			},
			min: 0.0,
			max: 0.5,
		},
		{
			name: "vector only",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Quality: TextQuality{Score: 0.8},
				},
			},
			min: 0.4,
			max: 0.6,
		},
		{
			name: "OCR only",
			result: &HybridResult{
				MergedRegions: []OCRRegion{
					{RecConfidence: 0.7},
				},
			},
			min: 0.2,
			max: 0.4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := processor.calculateOverallScore(tt.result)
			assert.GreaterOrEqual(t, score, tt.min)
			assert.LessOrEqual(t, score, tt.max)
			assert.LessOrEqual(t, score, 1.0)
		})
	}
}

func TestHybridProcessor_CalculateTextCoverage(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name   string
		result *HybridResult
		want   float64
	}{
		{
			name: "short text",
			result: &HybridResult{
				CombinedText: "Hello",
			},
			want: 0.005, // 5 / 1000
		},
		{
			name: "long text",
			result: &HybridResult{
				CombinedText: strings.Repeat("a", 2000),
			},
			want: 1.0, // capped at 1.0
		},
		{
			name: "empty text",
			result: &HybridResult{
				CombinedText: "",
			},
			want: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			coverage := processor.calculateTextCoverage(tt.result)
			assert.InDelta(t, tt.want, coverage, 0.001)
		})
	}
}

func TestHybridProcessor_CalculateAverageConfidence(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name   string
		result *HybridResult
		want   float64
	}{
		{
			name: "vector and OCR",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Quality: TextQuality{Score: 1.0},
				},
				MergedRegions: []OCRRegion{
					{RecConfidence: 0.8},
					{RecConfidence: 0.6},
				},
			},
			want: 0.8, // (1.0 + 0.8 + 0.6) / 3
		},
		{
			name:   "no sources",
			result: &HybridResult{},
			want:   0.0,
		},
		{
			name: "vector only",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Quality: TextQuality{Score: 0.9},
				},
			},
			want: 0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			confidence := processor.calculateAverageConfidence(tt.result)
			assert.InDelta(t, tt.want, confidence, 0.01)
		})
	}
}

func TestHybridProcessor_CalculateDuplicationLevel(t *testing.T) {
	processor := NewHybridProcessor(nil)

	tests := []struct {
		name   string
		result *HybridResult
		want   float64
	}{
		{
			name: "no duplication",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Text: "12345",
				},
				OCRResults: []ImageResult{
					{
						OCRRegions: []OCRRegion{
							{Text: "67890"},
						},
					},
				},
				CombinedText: "1234567890",
			},
			want: 0.0,
		},
		{
			name: "complete duplication",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Text: "12345",
				},
				OCRResults: []ImageResult{
					{
						OCRRegions: []OCRRegion{
							{Text: "12345"},
						},
					},
				},
				CombinedText: "12345",
			},
			want: 0.5, // compression ratio of 0.5
		},
		{
			name: "empty combined text",
			result: &HybridResult{
				VectorText: &TextExtraction{
					Text: "12345",
				},
				CombinedText: "",
			},
			want: 0.0, // When combined text is empty, duplication is 0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			duplication := processor.calculateDuplicationLevel(tt.result)
			assert.InDelta(t, tt.want, duplication, 0.1)
		})
	}
}

func TestHybridProcessor_MergeWithSmartStrategy(t *testing.T) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy:       MergeStrategyHybridSmart,
		ConfidenceThreshold: 0.7,
		PreferVectorText:    true,
		MinOCRConfidence:    0.3,
	})

	t.Run("high quality vector text", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "High quality vector",
			Quality:    TextQuality{HasText: true, Score: 0.9},
			Positions: []TextPosition{
				{Text: "High quality vector", X: 100, Y: 100, Width: 200, Height: 50},
			},
		}

		ocrResults := []ImageResult{
			{
				OCRRegions: []OCRRegion{
					{Text: "Additional OCR", RecConfidence: 0.8,
						Box: struct{ X, Y, W, H int }{X: 300, Y: 100, W: 150, H: 40}},
				},
			},
		}

		result, err := processor.MergeResults(vectorText, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEmpty(t, result.CombinedText)
		assert.Equal(t, "hybrid_smart", result.ProcessingInfo.MergeMethod)
	})

	t.Run("low quality vector text enhanced with OCR", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "Poor quality",
			Quality:    TextQuality{HasText: true, Score: 0.5},
			Positions: []TextPosition{
				{Text: "Poor quality", X: 100, Y: 100, Width: 100, Height: 30},
			},
		}

		ocrResults := []ImageResult{
			{
				OCRRegions: []OCRRegion{
					{Text: "High confidence OCR enhancement",
						RecConfidence: 0.95,
						Box:           struct{ X, Y, W, H int }{X: 300, Y: 100, W: 250, H: 40}},
				},
			},
		}

		result, err := processor.MergeResults(vectorText, ocrResults, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEmpty(t, result.CombinedText)
	})
}

// Benchmark tests.
func BenchmarkMergeResults_Append(b *testing.B) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy: MergeStrategyAppend,
	})

	vectorText := &TextExtraction{
		PageNumber: 1,
		Text:       "Vector text content",
		Quality:    TextQuality{HasText: true, Score: 0.9},
	}

	ocrResults := []ImageResult{
		{
			OCRRegions: []OCRRegion{
				{Text: "OCR text 1", RecConfidence: 0.85},
				{Text: "OCR text 2", RecConfidence: 0.75},
			},
		},
	}

	for range b.N {
		_, _ = processor.MergeResults(vectorText, ocrResults, 800, 600)
	}
}

func BenchmarkMergeResults_Spatial(b *testing.B) {
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy: MergeStrategySpatial,
	})

	vectorText := &TextExtraction{
		PageNumber: 1,
		Text:       "Vector text",
		Quality:    TextQuality{HasText: true, Score: 0.9},
		Positions: []TextPosition{
			{Text: "Line 1", X: 100, Y: 100, Width: 200, Height: 50},
			{Text: "Line 2", X: 100, Y: 150, Width: 200, Height: 50},
		},
	}

	ocrResults := []ImageResult{
		{
			OCRRegions: []OCRRegion{
				{Text: "OCR 1", RecConfidence: 0.85,
					Box: struct{ X, Y, W, H int }{X: 300, Y: 100, W: 150, H: 40}},
				{Text: "OCR 2", RecConfidence: 0.75,
					Box: struct{ X, Y, W, H int }{X: 300, Y: 150, W: 150, H: 40}},
			},
		},
	}

	for range b.N {
		_, _ = processor.MergeResults(vectorText, ocrResults, 800, 600)
	}
}

func BenchmarkTextSimilarity(b *testing.B) {
	processor := NewHybridProcessor(nil)
	text1 := "This is a sample text for similarity testing"
	text2 := "This is another sample text for comparison"

	for range b.N {
		_ = processor.textSimilarity(text1, text2)
	}
}

func BenchmarkElementsOverlap(b *testing.B) {
	processor := NewHybridProcessor(&HybridConfig{
		SpatialOverlapThreshold: 0.1,
	})

	elemA := TextElement{X: 100, Y: 100, Width: 100, Height: 100}
	elemB := TextElement{X: 120, Y: 120, Width: 100, Height: 100}

	for range b.N {
		_ = processor.elementsOverlap(elemA, elemB)
	}
}

func BenchmarkCalculateQualityMetrics(b *testing.B) {
	processor := NewHybridProcessor(nil)

	result := &HybridResult{
		VectorText: &TextExtraction{
			Text:    "Vector text content",
			Quality: TextQuality{Score: 0.9},
		},
		OCRResults: []ImageResult{
			{
				OCRRegions: []OCRRegion{
					{Text: "OCR text 1", RecConfidence: 0.85},
					{Text: "OCR text 2", RecConfidence: 0.75},
				},
			},
		},
		MergedRegions: []OCRRegion{
			{Text: "OCR text 1", RecConfidence: 0.85},
			{Text: "OCR text 2", RecConfidence: 0.75},
		},
		CombinedText: "Vector text content OCR text 1 OCR text 2",
	}

	for range b.N {
		processor.calculateQualityMetrics(result)
	}
}

// Edge case tests.
func TestHybridProcessor_EdgeCases(t *testing.T) {
	// Use config with deduplication disabled for edge case tests
	processor := NewHybridProcessor(&HybridConfig{
		MergeStrategy:        MergeStrategyAppend,
		DeduplicationEnabled: false,
		MinOCRConfidence:     0.0,
	})

	t.Run("very long combined text", func(t *testing.T) {
		longText := strings.Repeat("word ", 10000)
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       longText,
			Quality:    TextQuality{HasText: true, Score: 0.9},
		}

		result, err := processor.MergeResults(vectorText, nil, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEmpty(t, result.CombinedText)
	})

	t.Run("special characters in text", func(t *testing.T) {
		specialText := "Special chars: @#$%^&*(){}[]|\\:;<>?,./~`"
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       specialText,
			Quality:    TextQuality{HasText: true, Score: 0.9},
		}

		result, err := processor.MergeResults(vectorText, nil, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, specialText)
	})

	t.Run("unicode text", func(t *testing.T) {
		unicodeText := "Hello World Arabic English"
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       unicodeText,
			Quality:    TextQuality{HasText: true, Score: 0.9},
		}

		result, err := processor.MergeResults(vectorText, nil, 800, 600)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result.CombinedText, unicodeText)
	})

	t.Run("many OCR regions", func(t *testing.T) {
		var ocrRegions []OCRRegion
		for i := range 100 {
			ocrRegions = append(ocrRegions, OCRRegion{
				Text:          "Region " + string(rune('A'+i%26)),
				RecConfidence: 0.8,
				Box:           struct{ X, Y, W, H int }{X: i * 10, Y: i * 10, W: 100, H: 50},
			})
		}

		ocrResults := []ImageResult{{OCRRegions: ocrRegions}}

		result, err := processor.MergeResults(nil, ocrResults, 2000, 2000)

		require.NoError(t, err)
		require.NotNil(t, result)
		assert.NotEmpty(t, result.CombinedText)
	})

	t.Run("zero dimensions", func(t *testing.T) {
		vectorText := &TextExtraction{
			PageNumber: 1,
			Text:       "Test",
			Quality:    TextQuality{HasText: true, Score: 0.9},
		}

		result, err := processor.MergeResults(vectorText, nil, 0, 0)

		require.NoError(t, err)
		require.NotNil(t, result)
	})
}

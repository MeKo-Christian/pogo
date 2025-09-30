package pdf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProcessingStrategy_String(t *testing.T) {
	tests := []struct {
		name     string
		strategy ProcessingStrategy
		want     string
	}{
		{
			name:     "vector text strategy",
			strategy: StrategyVectorText,
			want:     "vector_text",
		},
		{
			name:     "ocr strategy",
			strategy: StrategyOCR,
			want:     "ocr",
		},
		{
			name:     "hybrid strategy",
			strategy: StrategyHybrid,
			want:     "hybrid",
		},
		{
			name:     "skip strategy",
			strategy: StrategySkip,
			want:     "skip",
		},
		{
			name:     "unknown strategy",
			strategy: ProcessingStrategy(999),
			want:     "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.strategy.String()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDefaultAnalyzerConfig(t *testing.T) {
	cfg := DefaultAnalyzerConfig()

	require.NotNil(t, cfg)
	assert.InDelta(t, 0.7, cfg.VectorTextQualityThreshold, 1e-9)
	assert.InDelta(t, 0.8, cfg.VectorTextCoverageThreshold, 1e-9)
	assert.True(t, cfg.HybridModeEnabled)
	assert.True(t, cfg.OCRFallbackEnabled)
	assert.InDelta(t, 0.1, cfg.MinTextDensityForOCR, 1e-9)
}

func TestNewPageAnalyzer(t *testing.T) {
	t.Run("with nil config", func(t *testing.T) {
		analyzer := NewPageAnalyzer(nil)

		require.NotNil(t, analyzer)
		assert.NotNil(t, analyzer.config)
		assert.NotNil(t, analyzer.textExtractor)
		assert.InDelta(t, DefaultAnalyzerConfig().VectorTextQualityThreshold,
			analyzer.config.VectorTextQualityThreshold, 1e-9)
	})

	t.Run("with custom config", func(t *testing.T) {
		customCfg := &AnalyzerConfig{
			VectorTextQualityThreshold:  0.6,
			VectorTextCoverageThreshold: 0.75,
			HybridModeEnabled:           false,
			OCRFallbackEnabled:          false,
			MinTextDensityForOCR:        0.2,
		}

		analyzer := NewPageAnalyzer(customCfg)

		require.NotNil(t, analyzer)
		assert.InDelta(t, 0.6, analyzer.config.VectorTextQualityThreshold, 1e-9)
		assert.InDelta(t, 0.75, analyzer.config.VectorTextCoverageThreshold, 1e-9)
		assert.False(t, analyzer.config.HybridModeEnabled)
		assert.False(t, analyzer.config.OCRFallbackEnabled)
		assert.InDelta(t, 0.2, analyzer.config.MinTextDensityForOCR, 1e-9)
	})
}

func TestPageAnalyzer_GetConfig(t *testing.T) {
	cfg := &AnalyzerConfig{
		VectorTextQualityThreshold: 0.5,
	}
	analyzer := NewPageAnalyzer(cfg)

	gotCfg := analyzer.GetConfig()
	assert.Equal(t, cfg, gotCfg)
	assert.InDelta(t, 0.5, gotCfg.VectorTextQualityThreshold, 1e-9)
}

func TestPageAnalyzer_UpdateConfig(t *testing.T) {
	analyzer := NewPageAnalyzer(nil)
	originalCfg := analyzer.GetConfig()

	t.Run("update with valid config", func(t *testing.T) {
		newCfg := &AnalyzerConfig{
			VectorTextQualityThreshold:  0.9,
			VectorTextCoverageThreshold: 0.95,
			HybridModeEnabled:           false,
			OCRFallbackEnabled:          false,
			MinTextDensityForOCR:        0.3,
		}

		analyzer.UpdateConfig(newCfg)

		assert.InDelta(t, 0.9, analyzer.config.VectorTextQualityThreshold, 1e-9)
		assert.InDelta(t, 0.95, analyzer.config.VectorTextCoverageThreshold, 1e-9)
		assert.False(t, analyzer.config.HybridModeEnabled)
		assert.False(t, analyzer.config.OCRFallbackEnabled)
	})

	t.Run("update with nil config", func(t *testing.T) {
		analyzer.UpdateConfig(originalCfg)
		analyzer.UpdateConfig(nil)

		// Config should remain unchanged
		assert.NotNil(t, analyzer.config)
	})
}

func TestPageAnalyzer_DetermineStrategy(t *testing.T) {
	tests := []struct {
		name           string
		config         *AnalyzerConfig
		extraction     *TextExtraction
		hasImages      bool
		imageCount     int
		wantStrategy   ProcessingStrategy
		wantScoreRange [2]float64 // min, max
		reasoningPart  string
	}{
		{
			name:           "no extraction, no images",
			config:         DefaultAnalyzerConfig(),
			extraction:     nil,
			hasImages:      false,
			imageCount:     0,
			wantStrategy:   StrategySkip,
			wantScoreRange: [2]float64{0.0, 0.2},
			reasoningPart:  "skipping",
		},
		{
			name:           "no extraction, has images",
			config:         DefaultAnalyzerConfig(),
			extraction:     nil,
			hasImages:      true,
			imageCount:     1,
			wantStrategy:   StrategyOCR,
			wantScoreRange: [2]float64{0.7, 0.9},
			reasoningPart:  "OCR",
		},
		{
			name:   "high quality vector text, no images",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.9,
					HasText: true,
				},
				Coverage: 0.85,
			},
			hasImages:      false,
			imageCount:     0,
			wantStrategy:   StrategyVectorText,
			wantScoreRange: [2]float64{0.9, 1.0},
			reasoningPart:  "vector text",
		},
		{
			name:   "high quality vector text with images (hybrid enabled)",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.9,
					HasText: true,
				},
				Coverage: 0.85,
			},
			hasImages:      true,
			imageCount:     2,
			wantStrategy:   StrategyHybrid,
			wantScoreRange: [2]float64{0.85, 1.0},
			reasoningPart:  "hybrid",
		},
		{
			name: "high quality vector text with images (hybrid disabled)",
			config: &AnalyzerConfig{
				VectorTextQualityThreshold:  0.7,
				VectorTextCoverageThreshold: 0.8,
				HybridModeEnabled:           false,
				OCRFallbackEnabled:          true,
				MinTextDensityForOCR:        0.1,
			},
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.9,
					HasText: true,
				},
				Coverage: 0.85,
			},
			hasImages:      true,
			imageCount:     1,
			wantStrategy:   StrategyVectorText,
			wantScoreRange: [2]float64{0.9, 1.0},
			reasoningPart:  "vector text",
		},
		{
			name:   "moderate quality vector text with images",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.6,
					HasText: true,
				},
				Coverage: 0.5,
			},
			hasImages:      true,
			imageCount:     1,
			wantStrategy:   StrategyHybrid,
			wantScoreRange: [2]float64{0.6, 0.8},
			reasoningPart:  "hybrid",
		},
		{
			name:   "moderate quality, no images, OCR fallback enabled",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.6,
					HasText: true,
				},
				Coverage: 0.5,
			},
			hasImages:      false,
			imageCount:     0,
			wantStrategy:   StrategyVectorText,
			wantScoreRange: [2]float64{0.4, 0.6},
			reasoningPart:  "vector text",
		},
		{
			name: "moderate quality with images, OCR fallback disabled",
			config: &AnalyzerConfig{
				VectorTextQualityThreshold:  0.7,
				VectorTextCoverageThreshold: 0.8,
				HybridModeEnabled:           true,
				OCRFallbackEnabled:          false,
				MinTextDensityForOCR:        0.1,
			},
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.6,
					HasText: true,
				},
				Coverage: 0.5,
			},
			hasImages:      true,
			imageCount:     1,
			wantStrategy:   StrategyHybrid,
			wantScoreRange: [2]float64{0.6, 0.8},
			reasoningPart:  "hybrid",
		},
		{
			name:   "poor quality, has images, OCR fallback enabled",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.3,
					HasText: true,
				},
				Coverage: 0.2,
			},
			hasImages:      true,
			imageCount:     1,
			wantStrategy:   StrategyOCR,
			wantScoreRange: [2]float64{0.6, 0.8},
			reasoningPart:  "OCR",
		},
		{
			name:   "poor quality, no images",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.3,
					HasText: true,
				},
				Coverage: 0.2,
			},
			hasImages:      false,
			imageCount:     0,
			wantStrategy:   StrategyVectorText,
			wantScoreRange: [2]float64{0.2, 0.4},
			reasoningPart:  "vector text",
		},
		{
			name:   "very poor quality, no text, no images",
			config: DefaultAnalyzerConfig(),
			extraction: &TextExtraction{
				Quality: TextQuality{
					Score:   0.1,
					HasText: false,
				},
				Coverage: 0.0,
			},
			hasImages:      false,
			imageCount:     0,
			wantStrategy:   StrategySkip,
			wantScoreRange: [2]float64{0.0, 0.2},
			reasoningPart:  "skipping",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analyzer := NewPageAnalyzer(tt.config)

			strategy, score, reasoning := analyzer.determineStrategy(tt.extraction, tt.hasImages, tt.imageCount)

			assert.Equal(t, tt.wantStrategy, strategy, "strategy mismatch")
			assert.GreaterOrEqual(t, score, tt.wantScoreRange[0], "score too low")
			assert.LessOrEqual(t, score, tt.wantScoreRange[1], "score too high")
			assert.Contains(t, reasoning, tt.reasoningPart, "reasoning doesn't contain expected substring")
		})
	}
}

func TestPageAnalyzer_EstimateOCRBenefit(t *testing.T) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())

	t.Run("no images", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.5,
				HasText: true,
			},
			Coverage: 0.5,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, false)
		assert.InDelta(t, 0.0, benefit, 1e-9)
	})

	t.Run("images but no extraction", func(t *testing.T) {
		benefit := analyzer.estimateOCRBenefit(nil, true)
		assert.InDelta(t, 1.0, benefit, 1e-9)
	})

	t.Run("images with high quality extraction", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.9,
				HasText: true,
			},
			Coverage: 0.95,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, true)
		// Quality gap = 0.1, coverage gap = 0.05, average = 0.075
		// But minimum benefit is 0.1
		assert.InDelta(t, 0.1, benefit, 1e-9)
	})

	t.Run("images with poor quality extraction", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.3,
				HasText: true,
			},
			Coverage: 0.2,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, true)
		// Quality gap = 0.7, coverage gap = 0.8, average = 0.75
		assert.InDelta(t, 0.75, benefit, 0.01)
	})

	t.Run("images with moderate quality extraction", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.6,
				HasText: true,
			},
			Coverage: 0.5,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, true)
		// Quality gap = 0.4, coverage gap = 0.5, average = 0.45
		assert.InDelta(t, 0.45, benefit, 0.01)
	})

	t.Run("images with no text but extraction object", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.0,
				HasText: false,
			},
			Coverage: 0.0,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, true)
		// Quality gap = 1.0, coverage gap = 1.0, average = 1.0
		assert.InDelta(t, 1.0, benefit, 1e-9)
	})
}

func TestPageAnalyzer_AnalyzeImages(t *testing.T) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())

	t.Run("non-existent file", func(t *testing.T) {
		hasImages, count := analyzer.analyzeImages("nonexistent.pdf", 1)
		assert.False(t, hasImages)
		assert.Equal(t, 0, count)
	})

	// Additional tests would require actual test PDF files
	// The current implementation relies on ExtractImages function
}

func TestPageAnalyzer_AnalyzePage(t *testing.T) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())

	t.Run("non-existent file", func(t *testing.T) {
		analysis, err := analyzer.AnalyzePage("nonexistent.pdf", 1)
		require.Error(t, err)
		assert.Nil(t, analysis)
		assert.Contains(t, err.Error(), "failed to extract text")
	})

	t.Run("invalid page number", func(t *testing.T) {
		// Using a non-existent file should still return an error
		analysis, err := analyzer.AnalyzePage("nonexistent.pdf", -1)
		require.Error(t, err)
		assert.Nil(t, analysis)
	})

	t.Run("zero page number", func(t *testing.T) {
		analysis, err := analyzer.AnalyzePage("nonexistent.pdf", 0)
		require.Error(t, err)
		assert.Nil(t, analysis)
	})
}

func TestPageAnalyzer_AnalyzePages(t *testing.T) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())

	t.Run("non-existent file", func(t *testing.T) {
		analyses, err := analyzer.AnalyzePages("nonexistent.pdf", "1")
		require.Error(t, err)
		assert.Nil(t, analyses)
		assert.Contains(t, err.Error(), "failed to extract text")
	})

	t.Run("invalid page range", func(t *testing.T) {
		analyses, err := analyzer.AnalyzePages("nonexistent.pdf", "abc")
		require.Error(t, err)
		assert.Nil(t, analyses)
		assert.Contains(t, err.Error(), "invalid page range")
	})

	t.Run("negative page range", func(t *testing.T) {
		analyses, err := analyzer.AnalyzePages("nonexistent.pdf", "-1")
		require.Error(t, err)
		assert.Nil(t, analyses)
		assert.Contains(t, err.Error(), "invalid page range")
	})

	t.Run("empty page range", func(t *testing.T) {
		analyses, err := analyzer.AnalyzePages("nonexistent.pdf", "")
		require.Error(t, err)
		assert.Nil(t, analyses)
	})

	t.Run("valid range format but non-existent file", func(t *testing.T) {
		analyses, err := analyzer.AnalyzePages("nonexistent.pdf", "1-5")
		require.Error(t, err)
		assert.Nil(t, analyses)
	})
}

func TestPageAnalysis_Structure(t *testing.T) {
	// Test that PageAnalysis struct can be properly constructed and accessed
	analysis := &PageAnalysis{
		PageNumber:          1,
		RecommendedStrategy: StrategyHybrid,
		VectorTextExtraction: &TextExtraction{
			Quality: TextQuality{
				Score:   0.8,
				HasText: true,
			},
			Coverage: 0.75,
		},
		HasImages:           true,
		ImageCount:          3,
		VectorTextQuality:   0.8,
		VectorTextCoverage:  0.75,
		EstimatedOCRBenefit: 0.3,
		AnalysisScore:       0.85,
		Reasoning:           "Test reasoning",
	}

	assert.Equal(t, 1, analysis.PageNumber)
	assert.Equal(t, StrategyHybrid, analysis.RecommendedStrategy)
	assert.NotNil(t, analysis.VectorTextExtraction)
	assert.True(t, analysis.HasImages)
	assert.Equal(t, 3, analysis.ImageCount)
	assert.InDelta(t, 0.8, analysis.VectorTextQuality, 1e-9)
	assert.InDelta(t, 0.75, analysis.VectorTextCoverage, 1e-9)
	assert.InDelta(t, 0.3, analysis.EstimatedOCRBenefit, 1e-9)
	assert.InDelta(t, 0.85, analysis.AnalysisScore, 1e-9)
	assert.Equal(t, "Test reasoning", analysis.Reasoning)
}

func TestAnalyzerConfig_Defaults(t *testing.T) {
	// Verify all default configuration values
	cfg := DefaultAnalyzerConfig()

	tests := []struct {
		name  string
		value interface{}
		want  interface{}
	}{
		{"VectorTextQualityThreshold", cfg.VectorTextQualityThreshold, 0.7},
		{"VectorTextCoverageThreshold", cfg.VectorTextCoverageThreshold, 0.8},
		{"HybridModeEnabled", cfg.HybridModeEnabled, true},
		{"OCRFallbackEnabled", cfg.OCRFallbackEnabled, true},
		{"MinTextDensityForOCR", cfg.MinTextDensityForOCR, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.value)
		})
	}
}

func TestPageAnalyzer_ConfigPropagation(t *testing.T) {
	// Test that config changes propagate to text extractor
	analyzer := NewPageAnalyzer(nil)

	// Initial threshold
	assert.InDelta(t, 0.7, analyzer.config.VectorTextQualityThreshold, 1e-9)

	// Update config
	newCfg := &AnalyzerConfig{
		VectorTextQualityThreshold:  0.5,
		VectorTextCoverageThreshold: 0.6,
		HybridModeEnabled:           false,
		OCRFallbackEnabled:          false,
		MinTextDensityForOCR:        0.2,
	}

	analyzer.UpdateConfig(newCfg)

	// Verify config was updated
	assert.InDelta(t, 0.5, analyzer.config.VectorTextQualityThreshold, 1e-9)

	// Note: We can't directly test if the text extractor threshold was updated
	// without exposing internal state, but the UpdateConfig method should handle it
}

func TestPageAnalyzer_EdgeCases(t *testing.T) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())

	t.Run("zero values in extraction", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.0,
				HasText: false,
			},
			Coverage: 0.0,
		}

		strategy, score, reasoning := analyzer.determineStrategy(extraction, false, 0)

		assert.Equal(t, StrategySkip, strategy)
		assert.GreaterOrEqual(t, score, 0.0)
		assert.NotEmpty(t, reasoning)
	})

	t.Run("exact threshold values", func(t *testing.T) {
		// Test with quality exactly at threshold
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.7, // Exactly at VectorTextQualityThreshold
				HasText: true,
			},
			Coverage: 0.8, // Exactly at VectorTextCoverageThreshold
		}

		strategy, score, reasoning := analyzer.determineStrategy(extraction, false, 0)

		assert.Equal(t, StrategyVectorText, strategy)
		assert.Greater(t, score, 0.8)
		assert.NotEmpty(t, reasoning)
	})

	t.Run("negative values protection", func(t *testing.T) {
		// The estimateOCRBenefit should handle edge cases gracefully
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   1.0,
				HasText: true,
			},
			Coverage: 1.0,
		}

		benefit := analyzer.estimateOCRBenefit(extraction, true)
		// Should return minimum benefit of 0.1
		assert.InDelta(t, 0.1, benefit, 1e-9)
	})

	t.Run("very large image count", func(t *testing.T) {
		extraction := &TextExtraction{
			Quality: TextQuality{
				Score:   0.5,
				HasText: true,
			},
			Coverage: 0.5,
		}

		strategy, score, reasoning := analyzer.determineStrategy(extraction, true, 1000)

		// Should still recommend hybrid with many images
		assert.Equal(t, StrategyHybrid, strategy)
		assert.NotEmpty(t, reasoning)
		assert.GreaterOrEqual(t, score, 0.0)
		assert.LessOrEqual(t, score, 1.0)
	})
}

// Benchmark tests for performance monitoring.
func BenchmarkDetermineStrategy(b *testing.B) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())
	extraction := &TextExtraction{
		Quality: TextQuality{
			Score:   0.8,
			HasText: true,
		},
		Coverage: 0.75,
	}

	b.ResetTimer()
	for range b.N {
		analyzer.determineStrategy(extraction, true, 5)
	}
}

func BenchmarkEstimateOCRBenefit(b *testing.B) {
	analyzer := NewPageAnalyzer(DefaultAnalyzerConfig())
	extraction := &TextExtraction{
		Quality: TextQuality{
			Score:   0.6,
			HasText: true,
		},
		Coverage: 0.5,
	}

	b.ResetTimer()
	for range b.N {
		analyzer.estimateOCRBenefit(extraction, true)
	}
}

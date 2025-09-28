package batch

import (
	"strconv"
	"strings"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/MeKo-Tech/pogo/internal/pipeline"
)

// buildPipeline creates an OCR pipeline from the batch configuration.
func buildPipeline(config *Config, progressCallback pipeline.ProgressCallback) (*pipeline.Pipeline, error) {
	b := pipeline.NewBuilder().
		WithModelsDir(config.ModelsDir).
		WithLanguage(config.Language).
		WithParallelWorkers(config.Workers).
		WithBatchSize(config.BatchSize).
		WithMemoryLimit(parseMemoryLimitOrDefault(config.MemoryLimitStr)).
		WithMaxGoroutines(config.MaxGoroutines).
		WithResourceThreshold(config.MemoryThreshold).
		WithAdaptiveScaling(config.AdaptiveScaling).
		WithBackpressure(config.Backpressure).
		WithProgressCallback(progressCallback)

	b = configurePipelineModels(b, config)
	b = configurePipelineFeatures(b, config)
	b = configurePipelineThresholds(b, config)

	return b.Build()
}

// configurePipelineModels sets up model-related configuration on the pipeline builder.
func configurePipelineModels(b *pipeline.Builder, config *Config) *pipeline.Builder {
	if config.DetModel != "" {
		b = b.WithDetectorModelPath(config.DetModel)
	}
	if config.RecModel != "" {
		b = b.WithRecognizerModelPath(config.RecModel)
	}
	if config.DictCSV != "" {
		parts := strings.Split(config.DictCSV, ",")
		b = b.WithDictionaryPaths(parts)
	}
	if config.DictLangs != "" {
		langs := strings.Split(config.DictLangs, ",")
		paths := models.GetDictionaryPathsForLanguages(config.ModelsDir, langs)
		if len(paths) > 0 {
			b = b.WithDictionaryPaths(paths)
		}
	}
	return b
}

// configurePipelineFeatures sets up feature-related configuration on the pipeline builder.
func configurePipelineFeatures(b *pipeline.Builder, config *Config) *pipeline.Builder {
	if config.DetectOrientation {
		b = b.WithOrientation(true)
	}
	if config.DetectTextline {
		b = b.WithTextLineOrientation(true)
	}
	if config.Rectify {
		b = b.WithRectification(true)
	}
	if config.RecHeight > 0 {
		b = b.WithImageHeight(config.RecHeight)
	}
	return b
}

// configurePipelineThresholds sets up threshold-related configuration on the pipeline builder.
func configurePipelineThresholds(b *pipeline.Builder, config *Config) *pipeline.Builder {
	b = b.WithDetectorThresholds(pipeline.DefaultConfig().Detector.DbThresh, float32(config.Confidence))
	if config.OrientThresh > 0 {
		b = b.WithOrientationThreshold(config.OrientThresh)
	}
	if config.TextlineThresh > 0 {
		b = b.WithTextLineOrientationThreshold(config.TextlineThresh)
	}
	if config.RectifyMask > 0 {
		b = b.WithRectifyMaskThreshold(config.RectifyMask)
	}
	if config.RectifyHeight > 0 {
		b = b.WithRectifyOutputHeight(config.RectifyHeight)
	}
	if config.RectifyDebugDir != "" {
		b = b.WithRectifyDebugDir(config.RectifyDebugDir)
	}
	if config.RectifyModel != "" {
		b = b.WithRectifyModelPath(config.RectifyModel)
	}
	return b
}

// parseMemoryLimitOrDefault parses memory limit or returns 0 if empty.
func parseMemoryLimitOrDefault(limitStr string) uint64 {
	if limitStr == "" {
		return 0
	}
	limit, err := parseMemoryLimit(limitStr)
	if err != nil {
		return 0
	}
	return limit
}

// parseMemoryLimit parses a memory limit string (e.g., "1GB", "512MB") into bytes.
func parseMemoryLimit(limit string) (uint64, error) {
	limit = strings.TrimSpace(strings.ToUpper(limit))

	multipliers := map[string]uint64{
		"B":  1,
		"KB": 1024,
		"MB": 1024 * 1024,
		"GB": 1024 * 1024 * 1024,
		"TB": 1024 * 1024 * 1024 * 1024,
	}

	for suffix, multiplier := range multipliers {
		if strings.HasSuffix(limit, suffix) {
			numStr := strings.TrimSuffix(limit, suffix)
			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return 0, err
			}
			return uint64(num * float64(multiplier)), nil
		}
	}

	// Try parsing as plain number (bytes)
	num, err := strconv.ParseUint(limit, 10, 64)
	return num, err
}

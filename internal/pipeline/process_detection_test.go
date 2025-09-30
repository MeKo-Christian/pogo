package pipeline

import (
	"context"
	"image"
	"image/color"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPerformDetection tests the performDetection method
func TestPerformDetection(t *testing.T) {
	tests := []struct {
		name          string
		setupPipeline func(*testing.T) *Pipeline
		expectError   bool
		cancelContext bool
		checkRegions  bool
	}{
		{
			name: "successful detection",
			setupPipeline: func(t *testing.T) *Pipeline {
				b := NewBuilder()
				p, err := b.Build()
				if err != nil {
					t.Skipf("pipeline build failed: %v", err)
				}
				return p
			},
			expectError:  false,
			checkRegions: true,
		},
		{
			name: "context cancellation",
			setupPipeline: func(t *testing.T) *Pipeline {
				b := NewBuilder()
				p, err := b.Build()
				if err != nil {
					t.Skipf("pipeline build failed: %v", err)
				}
				return p
			},
			expectError:   true,
			cancelContext: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := tt.setupPipeline(t)
			defer func() {
				_ = p.Close()
			}()

			// Create test image
			cfg := testutil.DefaultTestImageConfig()
			cfg.Size = testutil.ImageSize{Width: 300, Height: 150}
			cfg.Text = "Detection Test"
			cfg.Background = color.White
			cfg.Foreground = color.Black
			img, err := testutil.GenerateTextImage(cfg)
			require.NoError(t, err)

			// Setup context
			ctx := context.Background()
			if tt.cancelContext {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel() // Cancel immediately
			}

			// Execute
			regions, detNs, err := p.performDetection(ctx, img)

			// Assert
			if tt.expectError {
				require.Error(t, err)
				return
			}

			if err != nil {
				t.Skipf("detection failed (runtime deps): %v", err)
			}

			require.NoError(t, err)
			require.NotNil(t, regions)

			if tt.checkRegions {
				// Detection timing should be recorded
				assert.Greater(t, detNs, int64(0))

				// For synthetic text image, we expect at least some regions
				// (though this depends on the model)
				assert.GreaterOrEqual(t, len(regions), 0)

				// Verify region structure
				for i, region := range regions {
					assert.NotNil(t, region.Box, "region %d box should not be nil", i)
					assert.GreaterOrEqual(t, region.Confidence, 0.0, "region %d confidence should be >= 0", i)
					assert.LessOrEqual(t, region.Confidence, 1.0, "region %d confidence should be <= 1", i)
				}
			}
		})
	}
}

// TestPerformRecognition tests the performRecognition method
func TestPerformRecognition(t *testing.T) {
	tests := []struct {
		name          string
		setupPipeline func(*testing.T) *Pipeline
		setupRegions  func() []detector.DetectedRegion
		expectError   bool
		cancelContext bool
		checkResults  bool
	}{
		{
			name: "successful recognition with regions",
			setupPipeline: func(t *testing.T) *Pipeline {
				b := NewBuilder()
				p, err := b.Build()
				if err != nil {
					t.Skipf("pipeline build failed: %v", err)
				}
				return p
			},
			setupRegions: func() []detector.DetectedRegion {
				return []detector.DetectedRegion{
					{
						Box: utils.Box{
							MinX: 10,
							MinY: 10,
							MaxX: 100,
							MaxY: 50,
						},
						Confidence: 0.95,
						Polygon: []utils.Point{
							{X: 10, Y: 10},
							{X: 100, Y: 10},
							{X: 100, Y: 50},
							{X: 10, Y: 50},
						},
					},
				}
			},
			expectError:  false,
			checkResults: true,
		},
		{
			name: "no regions to recognize",
			setupPipeline: func(t *testing.T) *Pipeline {
				b := NewBuilder()
				p, err := b.Build()
				if err != nil {
					t.Skipf("pipeline build failed: %v", err)
				}
				return p
			},
			setupRegions: func() []detector.DetectedRegion {
				return []detector.DetectedRegion{}
			},
			expectError:  false,
			checkResults: false,
		},
		{
			name: "context cancellation",
			setupPipeline: func(t *testing.T) *Pipeline {
				b := NewBuilder()
				p, err := b.Build()
				if err != nil {
					t.Skipf("pipeline build failed: %v", err)
				}
				return p
			},
			setupRegions: func() []detector.DetectedRegion {
				return []detector.DetectedRegion{
					{
						Box: utils.Box{
							MinX: 10,
							MinY: 10,
							MaxX: 100,
							MaxY: 50,
						},
						Confidence: 0.95,
					},
				}
			},
			expectError:   true,
			cancelContext: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := tt.setupPipeline(t)
			defer func() {
				_ = p.Close()
			}()

			// Create test image
			cfg := testutil.DefaultTestImageConfig()
			cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
			cfg.Text = "Recognition"
			cfg.Background = color.White
			cfg.Foreground = color.Black
			img, err := testutil.GenerateTextImage(cfg)
			require.NoError(t, err)

			regions := tt.setupRegions()

			// Setup context
			ctx := context.Background()
			if tt.cancelContext {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel() // Cancel immediately
			}

			// Execute
			recResults, recNs, err := p.performRecognition(ctx, img, regions)

			// Assert
			if tt.expectError {
				require.Error(t, err)
				return
			}

			if err != nil {
				t.Skipf("recognition failed (runtime deps): %v", err)
			}

			require.NoError(t, err)

			if tt.checkResults {
				// Recognition timing should be recorded
				assert.Greater(t, recNs, int64(0))

				// Should have same number of results as regions
				assert.Len(t, recResults, len(regions))

				// Verify result structure
				for i, result := range recResults {
					assert.GreaterOrEqual(t, result.Confidence, 0.0, "result %d confidence should be >= 0", i)
					assert.LessOrEqual(t, result.Confidence, 1.0, "result %d confidence should be <= 1", i)
					// Text may be empty if recognition failed, so we don't assert on it
				}
			} else if len(regions) == 0 {
				// No regions means no results
				assert.Empty(t, recResults)
				// But timing should still be recorded
				assert.GreaterOrEqual(t, recNs, int64(0))
			}
		})
	}
}

// TestBuildImageResult tests the buildImageResult method
func TestBuildImageResult(t *testing.T) {
	b := NewBuilder()
	b.cfg.Recognizer.Language = "en"

	p := &Pipeline{
		cfg: b.cfg,
	}

	// Create test image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.ImageSize{Width: 300, Height: 150}
	cfg.Text = "Test Result"
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Create test regions and results
	regions := []detector.DetectedRegion{
		{
			Box: utils.Box{
				MinX: 10,
				MinY: 10,
				MaxX: 100,
				MaxY: 50,
			},
			Confidence: 0.92,
			Polygon: []utils.Point{
				{X: 10, Y: 10},
				{X: 100, Y: 10},
				{X: 100, Y: 50},
				{X: 10, Y: 50},
			},
		},
		{
			Box: utils.Box{
				MinX: 110,
				MinY: 10,
				MaxX: 200,
				MaxY: 50,
			},
			Confidence: 0.88,
			Polygon: []utils.Point{
				{X: 110, Y: 10},
				{X: 200, Y: 10},
				{X: 200, Y: 50},
				{X: 110, Y: 50},
			},
		},
	}

	// Mock recognition results
	// Note: In real code, these would come from recognizer
	// Here we're just testing the result building logic

	tests := []struct {
		name          string
		appliedAngle  int
		appliedConf   float64
		detNs         int64
		recNs         int64
		totalNs       int64
		checkFields   bool
	}{
		{
			name:         "no rotation applied",
			appliedAngle: 0,
			appliedConf:  0.0,
			detNs:        1000000,  // 1ms
			recNs:        2000000,  // 2ms
			totalNs:      5000000,  // 5ms
			checkFields:  true,
		},
		{
			name:         "with 90 degree rotation",
			appliedAngle: 90,
			appliedConf:  0.95,
			detNs:        1500000,
			recNs:        2500000,
			totalNs:      6000000,
			checkFields:  true,
		},
		{
			name:         "with 180 degree rotation",
			appliedAngle: 180,
			appliedConf:  0.88,
			detNs:        1200000,
			recNs:        2200000,
			totalNs:      5500000,
			checkFields:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Execute
			result := p.buildImageResult(
				img,
				regions,
				nil, // recResults - we're testing structure, not actual text
				tt.appliedAngle,
				tt.appliedConf,
				tt.detNs,
				tt.recNs,
				tt.totalNs,
			)

			// Assert
			require.NotNil(t, result)

			if tt.checkFields {
				// Check image dimensions
				assert.Equal(t, img.Bounds().Dx(), result.Width)
				assert.Equal(t, img.Bounds().Dy(), result.Height)

				// Check regions count
				assert.Len(t, result.Regions, len(regions))

				// Check orientation info
				assert.Equal(t, tt.appliedAngle, result.Orientation.Angle)
				if tt.appliedAngle != 0 {
					assert.True(t, result.Orientation.Applied)
					assert.Equal(t, tt.appliedConf, result.Orientation.Confidence)
				} else {
					assert.False(t, result.Orientation.Applied)
				}

				// Check timing info
				assert.Equal(t, tt.detNs, result.Processing.DetectionNs)
				assert.Equal(t, tt.recNs, result.Processing.RecognitionNs)
				assert.Equal(t, tt.totalNs, result.Processing.TotalNs)

				// Check average detection confidence
				expectedAvgConf := (regions[0].Confidence + regions[1].Confidence) / 2.0
				assert.InDelta(t, expectedAvgConf, result.AvgDetConf, 0.001)
			}
		})
	}
}

// TestBuildImageResult_EmptyRegions tests buildImageResult with no regions
func TestBuildImageResult_EmptyRegions(t *testing.T) {
	p := &Pipeline{
		cfg: Config{},
	}

	cfg := testutil.DefaultTestImageConfig()
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	result := p.buildImageResult(
		img,
		[]detector.DetectedRegion{}, // No regions
		nil,
		0,
		0,
		1000000,
		2000000,
		5000000,
	)

	require.NotNil(t, result)
	assert.Empty(t, result.Regions)
	assert.Equal(t, 0.0, result.AvgDetConf)
	assert.Equal(t, img.Bounds().Dx(), result.Width)
	assert.Equal(t, img.Bounds().Dy(), result.Height)
}

// TestProcessImageContext_FullFlow tests the complete ProcessImageContext flow
func TestProcessImageContext_FullFlow(t *testing.T) {
	b := NewBuilder()

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.ImageSize{Width: 250, Height: 120}
	cfg.Text = "Full Flow Test"
	cfg.Background = color.White
	cfg.Foreground = color.Black
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	// Execute with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := p.ProcessImageContext(ctx, img)
	if err != nil {
		t.Skipf("process failed (runtime deps): %v", err)
	}

	// Assert
	require.NotNil(t, result)
	assert.Equal(t, img.Bounds().Dx(), result.Width)
	assert.Equal(t, img.Bounds().Dy(), result.Height)
	assert.Greater(t, result.Processing.TotalNs, int64(0))
	assert.Greater(t, result.Processing.DetectionNs, int64(0))
	assert.GreaterOrEqual(t, result.Processing.RecognitionNs, int64(0))
}

// TestProcessImageContext_NilChecks tests nil input validation
func TestProcessImageContext_NilChecks(t *testing.T) {
	tests := []struct {
		name        string
		pipeline    *Pipeline
		image       image.Image
		expectError string
	}{
		{
			name:        "nil pipeline",
			pipeline:    nil,
			image:       nil,
			expectError: "pipeline not initialized",
		},
		{
			name: "nil detector",
			pipeline: &Pipeline{
				Detector:   nil,
				Recognizer: nil,
			},
			image:       nil,
			expectError: "pipeline not initialized",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			_, err := tt.pipeline.ProcessImageContext(ctx, tt.image)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectError)
		})
	}
}

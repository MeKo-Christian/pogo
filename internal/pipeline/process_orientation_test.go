package pipeline

import (
	"context"
	"image"
	"image/color"
	"testing"
	"time"

	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testText = "Test"

// TestApplyOrientationDetection tests the applyOrientationDetection method.
func TestApplyOrientationDetection(t *testing.T) {
	tests := []struct {
		name            string
		setupPipeline   func(*testing.T) *Pipeline
		expectedAngle   int
		expectError     bool
		expectedConfMin float64
		checkRotated    bool
		cancelContext   bool
	}{
		{
			name: "no orientation detection when disabled",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				return &Pipeline{
					cfg: Config{
						Orientation: orientation.Config{
							Enabled: false,
						},
					},
					Orienter: nil,
				}
			},
			expectedAngle: 0,
			expectError:   false,
			checkRotated:  false,
		},
		{
			name: "no orientation detection when orienter is nil",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				return &Pipeline{
					cfg: Config{
						Orientation: orientation.Config{
							Enabled: true,
						},
					},
					Orienter: nil,
				}
			},
			expectedAngle: 0,
			expectError:   false,
			checkRotated:  false,
		},
		{
			name: "context cancellation",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				cfg := orientation.DefaultConfig()
				cfg.Enabled = true
				cfg.HeuristicOnly = true // Use heuristic to avoid model deps

				classifier, err := orientation.NewClassifier(cfg)
				if err != nil {
					t.Skip("orientation classifier not available")
				}

				return &Pipeline{
					cfg: Config{
						Orientation: cfg,
					},
					Orienter: classifier,
				}
			},
			expectError:   true,
			cancelContext: true,
		},
		{
			name: "orientation detection with heuristic mode",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				cfg := orientation.DefaultConfig()
				cfg.Enabled = true
				cfg.HeuristicOnly = true

				classifier, err := orientation.NewClassifier(cfg)
				if err != nil {
					t.Skip("orientation classifier not available")
				}

				return &Pipeline{
					cfg: Config{
						Orientation: cfg,
					},
					Orienter: classifier,
				}
			},
			expectedAngle:   0, // Heuristic typically returns 0 for balanced images
			expectError:     false,
			expectedConfMin: 0.0,
			checkRotated:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := tt.setupPipeline(t)

			// Create test image
			cfg := testutil.DefaultTestImageConfig()
			cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
			cfg.Text = testText
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
			resultImg, angle, conf, err := p.applyOrientationDetection(ctx, img)

			// Assert
			if tt.expectError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, resultImg)

			if tt.checkRotated {
				assert.Equal(t, tt.expectedAngle, angle)
				if tt.expectedConfMin > 0 {
					assert.GreaterOrEqual(t, conf, tt.expectedConfMin)
				}
			}

			// Cleanup
			if p.Orienter != nil {
				p.Orienter.Close()
			}
		})
	}
}

// TestApplyOrientationRotation tests the applyOrientationRotation method.
func TestApplyOrientationRotation(t *testing.T) {
	// Create test image
	cfg := testutil.DefaultTestImageConfig()
	cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
	cfg.Text = testText
	img, err := testutil.GenerateTextImage(cfg)
	require.NoError(t, err)

	originalBounds := img.Bounds()

	p := &Pipeline{}

	tests := []struct {
		name        string
		angle       int
		checkWidth  int
		checkHeight int
	}{
		{
			name:        "no rotation (0 degrees)",
			angle:       0,
			checkWidth:  originalBounds.Dx(),
			checkHeight: originalBounds.Dy(),
		},
		{
			name:        "90 degree rotation",
			angle:       90,
			checkWidth:  originalBounds.Dy(), // Width becomes height
			checkHeight: originalBounds.Dx(), // Height becomes width
		},
		{
			name:        "180 degree rotation",
			angle:       180,
			checkWidth:  originalBounds.Dx(),
			checkHeight: originalBounds.Dy(),
		},
		{
			name:        "270 degree rotation",
			angle:       270,
			checkWidth:  originalBounds.Dy(), // Width becomes height
			checkHeight: originalBounds.Dx(), // Height becomes width
		},
		{
			name:        "invalid angle defaults to no rotation",
			angle:       45,
			checkWidth:  originalBounds.Dx(),
			checkHeight: originalBounds.Dy(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := p.applyOrientationRotation(img, tt.angle)
			require.NotNil(t, result)

			bounds := result.Bounds()
			assert.Equal(t, tt.checkWidth, bounds.Dx(), "width mismatch")
			assert.Equal(t, tt.checkHeight, bounds.Dy(), "height mismatch")
		})
	}
}

// TestPrepareOrientation tests the prepareOrientation method.
func TestPrepareOrientation(t *testing.T) {
	tests := []struct {
		name          string
		setupPipeline func(*testing.T) *Pipeline
		imageCount    int
		expectError   bool
		cancelContext bool
	}{
		{
			name: "no orientation processing when disabled",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				return &Pipeline{
					cfg: Config{
						Orientation: orientation.Config{
							Enabled: false,
						},
					},
					Orienter: nil,
				}
			},
			imageCount:  3,
			expectError: false,
		},
		{
			name: "no orientation processing when orienter is nil",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				return &Pipeline{
					cfg: Config{
						Orientation: orientation.Config{
							Enabled: true,
						},
					},
					Orienter: nil,
				}
			},
			imageCount:  3,
			expectError: false,
		},
		{
			name: "context cancellation before orientation",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				cfg := orientation.DefaultConfig()
				cfg.Enabled = true
				cfg.HeuristicOnly = true

				classifier, err := orientation.NewClassifier(cfg)
				if err != nil {
					t.Skip("orientation classifier not available")
				}

				return &Pipeline{
					cfg: Config{
						Orientation: cfg,
					},
					Orienter: classifier,
				}
			},
			imageCount:    2,
			expectError:   true,
			cancelContext: true,
		},
		{
			name: "batch orientation processing",
			setupPipeline: func(t *testing.T) *Pipeline {
				t.Helper()
				cfg := orientation.DefaultConfig()
				cfg.Enabled = true
				cfg.HeuristicOnly = true

				classifier, err := orientation.NewClassifier(cfg)
				if err != nil {
					t.Skip("orientation classifier not available")
				}

				return &Pipeline{
					cfg: Config{
						Orientation: cfg,
					},
					Orienter: classifier,
				}
			},
			imageCount:  3,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := tt.setupPipeline(t)
			defer func() {
				if p.Orienter != nil {
					p.Orienter.Close()
				}
			}()

			// Create test images
			images := make([]image.Image, tt.imageCount)
			for i := range tt.imageCount {
				cfg := testutil.DefaultTestImageConfig()
				cfg.Size = testutil.ImageSize{Width: 150, Height: 100}
				cfg.Text = testText
				img, err := testutil.GenerateTextImage(cfg)
				require.NoError(t, err)
				images[i] = img
			}

			// Setup context
			ctx := context.Background()
			if tt.cancelContext {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel() // Cancel immediately
			}

			// Execute
			orientationResults, workingImages, err := p.prepareOrientation(ctx, images)

			// Assert
			if tt.expectError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, orientationResults)
			require.NotNil(t, workingImages)
			assert.Len(t, orientationResults, tt.imageCount)
			assert.Len(t, workingImages, tt.imageCount)

			// Verify working images are valid
			for _, img := range workingImages {
				require.NotNil(t, img)
			}
		})
	}
}

// TestProcessImagesContext_WithOrientation tests ProcessImagesContext with orientation.
func TestProcessImagesContext_WithOrientation(t *testing.T) {
	// This test requires models, so we'll skip if not available
	b := NewBuilder()

	// Enable orientation with heuristic mode
	b.cfg.Orientation.Enabled = true
	b.cfg.Orientation.HeuristicOnly = true

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test images
	images := make([]image.Image, 2)
	for i := range images {
		cfg := testutil.DefaultTestImageConfig()
		cfg.Size = testutil.ImageSize{Width: 200, Height: 100}
		cfg.Text = "Hello"
		img, err := testutil.GenerateTextImage(cfg)
		require.NoError(t, err)
		images[i] = img
	}

	// Execute with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results, err := p.ProcessImagesContext(ctx, images)
	if err != nil {
		t.Skipf("process failed (runtime deps): %v", err)
	}

	require.NotNil(t, results)
	assert.Len(t, results, 2)

	// Verify orientation information is present
	for i, result := range results {
		assert.NotNil(t, result, "result %d should not be nil", i)
		// Orientation fields should be set even if angle is 0
		assert.GreaterOrEqual(t, result.Orientation.Angle, 0)
		assert.LessOrEqual(t, result.Orientation.Angle, 270)
	}
}

// TestProcessImagesContext_Cancellation tests context cancellation during processing.
func TestProcessImagesContext_Cancellation(t *testing.T) {
	b := NewBuilder()

	p, err := b.Build()
	if err != nil {
		t.Skipf("pipeline build failed: %v", err)
	}
	defer func() {
		require.NoError(t, p.Close())
	}()

	// Create test images
	images := make([]image.Image, 2)
	for i := range images {
		cfg := testutil.DefaultTestImageConfig()
		cfg.Text = testText
		img, err := testutil.GenerateTextImage(cfg)
		require.NoError(t, err)
		images[i] = img
	}

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Execute - should fail quickly due to cancellation
	_, err = p.ProcessImagesContext(ctx, images)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context")
}

package detector

import (
	"os"
	"testing"

	"github.com/MeKo-Tech/pogo/internal/models"
	"github.com/stretchr/testify/require"
)

func TestDetector_Warmup_SkipIfNoModel(t *testing.T) {
	modelPath := models.GetDetectionModelPath("", false)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Detection model not available, skipping warmup test")
	}
	cfg := DefaultConfig()
	cfg.ModelPath = modelPath
	det, err := NewDetector(cfg)
	require.NoError(t, err)
	defer func() { _ = det.Close() }()

	// Zero/no-op iterations should succeed
	require.NoError(t, det.Warmup(0))
	// One or two iterations should also succeed
	require.NoError(t, det.Warmup(2))
}

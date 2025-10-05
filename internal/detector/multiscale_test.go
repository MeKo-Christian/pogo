package detector

import (
	"testing"

	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/stretchr/testify/assert"
)

// helper to build a DetectedRegion with given box and confidence
func makeRegion(x1, y1, x2, y2 float64, conf float64) DetectedRegion {
	return DetectedRegion{
		Polygon:    []utils.Point{{X: x1, Y: y1}, {X: x2, Y: y1}, {X: x2, Y: y2}, {X: x1, Y: y2}},
		Box:        utils.NewBox(x1, y1, x2, y2),
		Confidence: conf,
	}
}

func TestMergeMultiScaleRegions_HardNMS(t *testing.T) {
	// Two overlapping boxes with IoU ~0.47; expect 1 kept with hard NMS at threshold 0.3
	r1 := makeRegion(10, 10, 50, 50, 0.9)
	r2 := makeRegion(18, 18, 58, 58, 0.8) // ~8px offset -> IoU ~0.47
	regs := []DetectedRegion{r1, r2}

	cfg := DefaultConfig()
	cfg.NMSMethod = "hard"
	cfg.MultiScale.Enabled = true
	cfg.MultiScale.MergeIoU = 0.3

	merged := mergeMultiScaleRegions(regs, cfg)
	assert.Len(t, merged, 1)
	// Highest confidence should be kept
	assert.InDelta(t, 0.9, merged[0].Confidence, 1e-6)
}

func TestMergeMultiScaleRegions_SoftNMS_Linear(t *testing.T) {
	// Overlap triggers linear soft-NMS decay; both remain above score threshold
	r1 := makeRegion(10, 10, 50, 50, 0.9)
	r2 := makeRegion(18, 18, 58, 58, 0.8) // ensure IoU >= threshold for decay
	regs := []DetectedRegion{r1, r2}

	cfg := DefaultConfig()
	cfg.NMSMethod = "linear"
	cfg.SoftNMSThresh = 0.1
	cfg.MultiScale.Enabled = true
	cfg.MultiScale.MergeIoU = 0.3

	merged := mergeMultiScaleRegions(regs, cfg)
	assert.Len(t, merged, 2)
	// Confidence of the lower one should decay but stay > SoftNMSThresh
	// merged is sorted desc; find the smaller confidence
	cmin := merged[1].Confidence
	assert.Greater(t, cmin, 0.1)
	assert.Less(t, cmin, 0.8)
}

func TestMergeMultiScaleRegions_FallbackIoU_FromNMSThreshold(t *testing.T) {
	// MergeIoU <= 0 should fallback to cfg.NMSThreshold
	// With IoU ~0.35 and NMSThreshold 0.4, hard NMS keeps both
	r1 := makeRegion(10, 10, 50, 50, 0.9)
	// Adjust to get smaller overlap; boxes: r1 area 1600, r3 area 1600, overlap ~400 => IoU ~ 400/(1600+1600-400)=400/2800=0.142 -> too low.
	// Use closer boxes to target ~0.35
	r3 := makeRegion(26, 26, 66, 66, 0.7)
	regs := []DetectedRegion{r1, r3}

	// Compute IoU to validate condition (not strictly required)
	iou := ComputeRegionIoU(r1.Box, r3.Box)
	assert.InDelta(t, 0.35, iou, 0.2) // accept a band around 0.35

	cfg := DefaultConfig()
	cfg.NMSMethod = "hard"
	cfg.NMSThreshold = 0.4
	cfg.MultiScale.Enabled = true
	cfg.MultiScale.MergeIoU = 0.0 // force fallback

	merged := mergeMultiScaleRegions(regs, cfg)
	assert.Len(t, merged, 2)
}

func TestMergeMultiScaleRegions_AdaptiveAndSizeAware_DoNotPanic(t *testing.T) {
	r1 := makeRegion(10, 10, 50, 50, 0.9)
	r2 := makeRegion(30, 30, 70, 70, 0.8)
	regs := []DetectedRegion{r1, r2}

	// Adaptive NMS
	cfgA := DefaultConfig()
	cfgA.NMSMethod = "hard"
	cfgA.UseAdaptiveNMS = true
	cfgA.NMSThreshold = 0.3
	cfgA.AdaptiveNMSScale = 1.0
	cfgA.MultiScale.Enabled = true
	mergedA := mergeMultiScaleRegions(regs, cfgA)
	assert.GreaterOrEqual(t, len(mergedA), 1)
	assert.LessOrEqual(t, len(mergedA), 2)

	// Size-aware NMS
	cfgS := DefaultConfig()
	cfgS.NMSMethod = "hard"
	cfgS.SizeAwareNMS = true
	cfgS.NMSThreshold = 0.3
	cfgS.SizeNMSScaleFactor = 0.1
	cfgS.MinRegionSize = 16
	cfgS.MaxRegionSize = 2048
	cfgS.MultiScale.Enabled = true
	mergedS := mergeMultiScaleRegions(regs, cfgS)
	assert.GreaterOrEqual(t, len(mergedS), 1)
	assert.LessOrEqual(t, len(mergedS), 2)
}

func TestGenerateAdaptiveScales_LargeImage(t *testing.T) {
	ms := DefaultMultiScaleConfig()
	ms.Adaptive = true
	ms.MaxLevels = 5
	ms.MinSide = 320
	scales := generateAdaptiveScales(4000, 3000, ms)
	assert.GreaterOrEqual(t, len(scales), 2)
	assert.InDelta(t, 1.0, scales[0], 1e-9)
	// Ensure not exceeding MaxLevels
	assert.LessOrEqual(t, len(scales), ms.MaxLevels)
}

func TestGenerateAdaptiveScales_SmallImage(t *testing.T) {
	ms := DefaultMultiScaleConfig()
	ms.Adaptive = true
	ms.MaxLevels = 5
	ms.MinSide = 256
	scales := generateAdaptiveScales(600, 400, ms)
	assert.GreaterOrEqual(t, len(scales), 1)
	assert.InDelta(t, 1.0, scales[0], 1e-9)
}

func TestIncrementalMergeEquivalence(t *testing.T) {
	// Prepare two sets that overlap across scales
	a1 := makeRegion(10, 10, 60, 60, 0.9)
	a2 := makeRegion(100, 100, 140, 140, 0.85)
	b1 := makeRegion(30, 30, 70, 70, 0.8)     // overlaps with a1
	b2 := makeRegion(160, 160, 200, 200, 0.7) // separate

	cfg := DefaultConfig()
	cfg.NMSMethod = "hard"
	cfg.NMSThreshold = 0.3
	cfg.MultiScale.Enabled = true

	// One-shot merge of union
	all := []DetectedRegion{a1, a2, b1, b2}
	final := mergeMultiScaleRegions(all, cfg)

	// Incremental merge
	inc := []DetectedRegion{}
	inc = append(inc, a1, a2)
	inc = mergeMultiScaleRegions(inc, cfg)
	inc = append(inc, b1, b2)
	inc = mergeMultiScaleRegions(inc, cfg)

	// Both strategies should converge to same or subset superset sizes depending on equal confidences
	assert.LessOrEqual(t, len(inc), len(final))
	assert.GreaterOrEqual(t, len(inc), len(final)-1)
}

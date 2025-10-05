package detector

import (
	"math"
	"testing"
)

// buildProbMap creates a WxH map filled with v and optionally overrides indices.
func buildProbMap(w, h int, v float32, overrides map[int]float32) []float32 {
	prob := make([]float32, w*h)
	for i := range prob {
		prob[i] = v
	}
	for i, val := range overrides {
		if i >= 0 && i < len(prob) {
			prob[i] = val
		}
	}
	return prob
}

func TestConfidenceMethods_MeanAndMax(t *testing.T) {
	// 2x2 region: values {0.2, 0.8, 0.6, 0.4} => mean=0.5, max=0.8
	w, h := 2, 2
	prob := []float32{0.2, 0.8, 0.6, 0.4}

	// Low threshold so all pixels are included into one component
	optsMean := PostProcessOptions{UseMinAreaRect: false, ConfidenceMethod: "mean"}
	regsMean := PostProcessDBWithOptions(prob, w, h, 0.1, 0.0, optsMean)
	if len(regsMean) != 1 {
		t.Fatalf("expected 1 region for mean, got %d", len(regsMean))
	}
	if math.Abs(regsMean[0].Confidence-0.5) > 1e-6 {
		t.Errorf("mean: expected 0.5, got %f", regsMean[0].Confidence)
	}

	optsMax := PostProcessOptions{UseMinAreaRect: false, ConfidenceMethod: "max"}
	regsMax := PostProcessDBWithOptions(prob, w, h, 0.1, 0.0, optsMax)
	if len(regsMax) != 1 {
		t.Fatalf("expected 1 region for max, got %d", len(regsMax))
	}
	if math.Abs(regsMax[0].Confidence-0.8) > 1e-6 {
		t.Errorf("max: expected 0.8, got %f", regsMax[0].Confidence)
	}
}

func TestConfidenceMethod_MeanVarAdjusted(t *testing.T) {
	// Mixed values to yield non-zero variance; check penalty applied
	w, h := 2, 2
	prob := []float32{0.2, 0.8, 0.6, 0.4} // mean=0.5
	opts := PostProcessOptions{UseMinAreaRect: false, ConfidenceMethod: "mean_var"}
	regs := PostProcessDBWithOptions(prob, w, h, 0.1, 0.0, opts)
	if len(regs) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regs))
	}
	if !(regs[0].Confidence <= 0.5 && regs[0].Confidence >= 0.0) {
		t.Errorf("mean_var should be within [0,0.5], got %f", regs[0].Confidence)
	}
}

func TestConfidenceCalibration_Gamma(t *testing.T) {
	// Uniform map with mean 0.5, apply gamma=2 => 0.25
	w, h := 4, 4
	prob := buildProbMap(w, h, 0.5, nil)
	opts := PostProcessOptions{UseMinAreaRect: true, ConfidenceMethod: "mean", CalibrationGamma: 2.0}
	regs := PostProcessDBWithOptions(prob, w, h, 0.1, 0.0, opts)
	if len(regs) != 1 {
		t.Fatalf("expected 1 region, got %d", len(regs))
	}
	if math.Abs(regs[0].Confidence-0.25) > 1e-6 {
		t.Errorf("gamma calibration expected 0.25, got %f", regs[0].Confidence)
	}
}

func TestAdaptiveConfidenceThresholding_FavorsSmallRegions(t *testing.T) {
	// Construct map with two regions: one small high-ish conf, one large slightly lower
	// Base threshold 0.6; without adaptive, only small high conf passes; with adaptive, small region can pass at slightly lower conf
	w, h := 40, 20
	prob := make([]float32, w*h)
	// Large region ~20x10 with value 0.58 (below base threshold)
	for y := 2; y < 12; y++ {
		for x := 2; x < 22; x++ {
			prob[y*w+x] = 0.58
		}
	}
	// Small region 2x2 with value 0.59 (just below base threshold)
	for y := 14; y < 16; y++ {
		for x := 30; x < 32; x++ {
			prob[y*w+x] = 0.59
		}
	}

	// DB threshold low so both become components
	baseThresh := float32(0.5)
	boxMin := float32(0.6)

	// Without adaptive: none should pass (both below 0.6)
	regsNoAdapt := PostProcessDBWithOptions(prob, w, h, baseThresh, boxMin, PostProcessOptions{UseMinAreaRect: true})
	if len(regsNoAdapt) != 0 {
		t.Fatalf("expected 0 regions without adaptive, got %d", len(regsNoAdapt))
	}

	// With adaptive: small region should pass due to lower effective threshold
	regsAdapt := PostProcessDBWithOptions(prob, w, h, baseThresh, boxMin, PostProcessOptions{UseMinAreaRect: true, AdaptiveConfidence: true, AdaptiveConfidenceScale: 0.5})
	if len(regsAdapt) < 1 {
		t.Fatalf("expected at least 1 region with adaptive, got %d", len(regsAdapt))
	}
}

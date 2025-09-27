package pipeline

import (
	"sync/atomic"
)

// Profiler aggregates simple counters/timers across multiple runs.
type Profiler struct {
	DetectionTimeNs   atomic.Int64
	RecognitionTimeNs atomic.Int64
	ImagesProcessed   atomic.Int64
	RegionsRecognized atomic.Int64
}

func (p *Profiler) Record(detNs, recNs int64, regions int) {
	p.DetectionTimeNs.Add(detNs)
	p.RecognitionTimeNs.Add(recNs)
	p.ImagesProcessed.Add(1)
	p.RegionsRecognized.Add(int64(regions))
}

// Snapshot returns cumulative metrics in milliseconds for readability.
func (p *Profiler) Snapshot() map[string]any {
	imgs := p.ImagesProcessed.Load()
	det := p.DetectionTimeNs.Load()
	rec := p.RecognitionTimeNs.Load()
	regs := p.RegionsRecognized.Load()
	out := map[string]any{
		"images":       imgs,
		"regions":      regs,
		"det_ms_total": det / 1_000_000,
		"rec_ms_total": rec / 1_000_000,
	}
	if imgs > 0 {
		out["det_ms_per_image"] = float64(det) / 1_000_000.0 / float64(imgs)
		out["rec_ms_per_image"] = float64(rec) / 1_000_000.0 / float64(imgs)
	}
	return out
}

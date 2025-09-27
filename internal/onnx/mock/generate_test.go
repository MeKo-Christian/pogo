package mock

import (
	"testing"
)

func TestNewUniformMap(t *testing.T) {
	m := NewUniformMap(10, 5, 0.7)
	if m.Width != 10 || m.Height != 5 {
		t.Fatalf("unexpected size: %dx%d", m.Width, m.Height)
	}
	if len(m.Data) != 50 {
		t.Fatalf("unexpected data len: %d", len(m.Data))
	}
	for _, v := range m.Data {
		if v < 0 || v > 1 {
			t.Fatalf("out of range: %f", v)
		}
	}
}

func TestNewCenteredBlobMap(t *testing.T) {
	m := NewCenteredBlobMap(9, 7, 1.0, 2.0)
	if len(m.Data) != m.Width*m.Height || m.Width != 9 || m.Height != 7 {
		t.Fatalf("unexpected shape or len")
	}
	// Center should be highest
	cx := m.Width / 2
	cy := m.Height / 2
	center := m.Data[cy*m.Width+cx]
	for i, v := range m.Data {
		if v > center+1e-6 {
			t.Fatalf("index %d > center: %f > %f", i, v, center)
		}
		if v < 0 || v > 1 {
			t.Fatalf("out of [0,1]: %f", v)
		}
	}
}

func TestNewTextStripeMap(t *testing.T) {
	m := NewTextStripeMap(8, 8, 2, 2, 0.9, 0.1)
	if len(m.Data) != 64 {
		t.Fatalf("len=%d", len(m.Data))
	}
	period := 4
	for y := range m.Height {
		want := float32(0.1)
		if (y % period) < 2 {
			want = 0.9
		}
		for x := range m.Width {
			if m.Data[y*m.Width+x] != want {
				t.Fatalf("stripe mismatch at row %d", y)
			}
		}
	}
}

func TestNewGreedyPathLogits_TxC(t *testing.T) {
	idx := []int{0, 3, 2, 2, 0}
	l := NewGreedyPathLogits(idx, 5, false, 0.9, 0.05)
	if len(l.Shape) != 3 || l.Shape[0] != 1 || l.Shape[1] != int64(len(idx)) || l.Shape[2] != 5 {
		t.Fatalf("bad shape: %v", l.Shape)
	}
	// For each timestep, verify argmax equals idx[t]
	for tStep := range idx {
		base := tStep * 5
		maxV := float32(-1)
		maxI := -1
		for c := range 5 {
			v := l.Data[base+c]
			if v > maxV {
				maxV = v
				maxI = c
			}
		}
		if maxI != idx[tStep] {
			t.Fatalf("t=%d want %d got %d", tStep, idx[tStep], maxI)
		}
	}
}

func TestNewGreedyPathLogits_CxT(t *testing.T) {
	idx := []int{1, 1, 4}
	l := NewGreedyPathLogits(idx, 6, true, 1.0, 0.0)
	if len(l.Shape) != 3 || l.Shape[0] != 1 || l.Shape[1] != 6 || l.Shape[2] != int64(len(idx)) {
		t.Fatalf("bad shape: %v", l.Shape)
	}
	// For each timestep, verify argmax equals idx[t]
	for tStep := range idx {
		maxV := float32(-1)
		maxI := -1
		for c := range 6 {
			v := l.Data[c*len(idx)+tStep]
			if v > maxV {
				maxV = v
				maxI = c
			}
		}
		if maxI != idx[tStep] {
			t.Fatalf("t=%d want %d got %d", tStep, idx[tStep], maxI)
		}
	}
}

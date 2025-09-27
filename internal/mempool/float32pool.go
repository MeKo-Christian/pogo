package mempool

import (
	"sync"
)

// A simple sized pool for []float32 buffers to reduce allocations on hot paths.

var pools sync.Map // key: size class (int), value: *sync.Pool

// sizeClass rounds n up to the next power-of-two-ish bucket to reduce churn.
func sizeClass(n int) int {
	if n <= 1024 {
		return 1024
	}
	// round up to next multiple of 1024
	const step = 1024
	r := (n + step - 1) / step
	return r * step
}

func GetFloat32(n int) []float32 {
	cls := sizeClass(n)
	pAny, _ := pools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]float32, cls) }})
	p, ok := pAny.(*sync.Pool)
	if !ok {
		// Fallback
		buf := make([]float32, cls)
		return buf[:n]
	}
	bufAny := p.Get()
	buf, ok := bufAny.([]float32)
	if !ok {
		buf = make([]float32, cls)
	}
	return buf[:n]
}

// PutFloat32 returns a buffer to the pool. It is safe to pass a nil slice.
func PutFloat32(buf []float32) {
	if buf == nil {
		return
	}
	cls := sizeClass(cap(buf))
	pAny, _ := pools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]float32, cls) }})
	p, ok := pAny.(*sync.Pool)
	if !ok {
		return // skip
	}
	// Reset length to full cap to avoid keeping len from caller; contents need not be zeroed.
	p.Put(buf[:cap(buf)])
}

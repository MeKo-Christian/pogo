package mempool

import (
	"sync"
)

// A simple sized pool for []float32 and []bool buffers to reduce allocations on hot paths.

var (
	float32Pools sync.Map // key: size class (int), value: *sync.Pool
	boolPools    sync.Map // key: size class (int), value: *sync.Pool
)

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

// GetFloat32 retrieves a []float32 buffer of at least n elements from the pool.
// The returned slice has length n but may have larger capacity.
// The caller must return it via PutFloat32 when done.
func GetFloat32(n int) []float32 {
	cls := sizeClass(n)
	pAny, _ := float32Pools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]float32, cls) }})
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
	// Ensure buffer has adequate capacity and reset length to full capacity
	if cap(buf) < cls {
		buf = make([]float32, cls)
	} else {
		buf = buf[:cap(buf)]
	}
	return buf[:n]
}

// PutFloat32 returns a buffer to the pool. It is safe to pass a nil slice.
func PutFloat32(buf []float32) {
	if buf == nil {
		return
	}
	cls := sizeClass(cap(buf))
	pAny, _ := float32Pools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]float32, cls) }})
	p, ok := pAny.(*sync.Pool)
	if !ok {
		return // skip
	}
	// Reset length to full cap to avoid keeping len from caller; contents need not be zeroed.
	p.Put(buf[:cap(buf)]) //nolint:staticcheck
}

// GetBool retrieves a []bool buffer of at least n elements from the pool.
// The returned slice has length n but may have larger capacity.
// The caller must return it via PutBool when done.
func GetBool(n int) []bool {
	cls := sizeClass(n)
	pAny, _ := boolPools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]bool, cls) }})
	p, ok := pAny.(*sync.Pool)
	if !ok {
		// Fallback
		buf := make([]bool, cls)
		return buf[:n]
	}
	bufAny := p.Get()
	buf, ok := bufAny.([]bool)
	if !ok {
		buf = make([]bool, cls)
	}
	// Ensure buffer has adequate capacity and reset length to full capacity
	if cap(buf) < cls {
		buf = make([]bool, cls)
	} else {
		buf = buf[:cap(buf)]
	}
	// Zero out the buffer since bool pools are reused and we need clean state
	for i := range buf[:n] {
		buf[i] = false
	}
	return buf[:n]
}

// PutBool returns a buffer to the pool. It is safe to pass a nil slice.
func PutBool(buf []bool) {
	if buf == nil {
		return
	}
	cls := sizeClass(cap(buf))
	pAny, _ := boolPools.LoadOrStore(cls, &sync.Pool{New: func() any { return make([]bool, cls) }})
	p, ok := pAny.(*sync.Pool)
	if !ok {
		return // skip
	}
	// Reset length to full cap to avoid keeping len from caller
	p.Put(buf[:cap(buf)]) //nolint:staticcheck
}

// GetFloat32Multiple retrieves multiple float32 buffers with the specified sizes.
// This is more efficient than calling GetFloat32 multiple times.
func GetFloat32Multiple(sizes []int) [][]float32 {
	if len(sizes) == 0 {
		return nil
	}
	buffers := make([][]float32, len(sizes))
	for i, size := range sizes {
		buffers[i] = GetFloat32(size)
	}
	return buffers
}

// PutFloat32Multiple returns multiple buffers to the pool.
// It is safe to pass nil slices in the array.
func PutFloat32Multiple(bufs [][]float32) {
	for _, buf := range bufs {
		PutFloat32(buf)
	}
}

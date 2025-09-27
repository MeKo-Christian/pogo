package pipeline

import (
	"runtime"
)

// MemStats summarizes memory usage information.
type MemStats struct {
	AllocBytes      uint64 `json:"alloc_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	SysBytes        uint64 `json:"sys_bytes"`
	NumGC           uint32 `json:"num_gc"`
	Goroutines      int    `json:"goroutines"`
}

// GetMemStats captures current memory statistics.
func GetMemStats() MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemStats{
		AllocBytes:      m.Alloc,
		TotalAllocBytes: m.TotalAlloc,
		SysBytes:        m.Sys,
		NumGC:           m.NumGC,
		Goroutines:      runtime.NumGoroutine(),
	}
}

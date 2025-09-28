// Package common provides shared utilities for benchmarking.
package common

import (
	"fmt"
	"runtime"
	"time"
)

// MemoryStats holds memory statistics for benchmarking.
type MemoryStats struct {
	Alloc         uint64
	TotalAlloc    uint64
	Sys           uint64
	Lookups       uint64
	Mallocs       uint64
	Frees         uint64
	HeapAlloc     uint64
	HeapSys       uint64
	HeapIdle      uint64
	HeapInuse     uint64
	HeapReleased  uint64
	HeapObjects   uint64
	StackInuse    uint64
	StackSys      uint64
	GCSys         uint64
	NextGC        uint64
	LastGC        uint64 // nanoseconds since program start
	NumGC         uint32
	NumForcedGC   uint32
	GCCPUFraction float64
}

// GetMemoryStats returns current memory statistics.
func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemoryStats{
		Alloc:         m.Alloc,
		TotalAlloc:    m.TotalAlloc,
		Sys:           m.Sys,
		Lookups:       m.Lookups,
		Mallocs:       m.Mallocs,
		Frees:         m.Frees,
		HeapAlloc:     m.HeapAlloc,
		HeapSys:       m.HeapSys,
		HeapIdle:      m.HeapIdle,
		HeapInuse:     m.HeapInuse,
		HeapReleased:  m.HeapReleased,
		HeapObjects:   m.HeapObjects,
		StackInuse:    m.StackInuse,
		StackSys:      m.StackSys,
		GCSys:         m.GCSys,
		NextGC:        m.NextGC,
		LastGC:        m.LastGC,
		NumGC:         m.NumGC,
		NumForcedGC:   m.NumForcedGC,
		GCCPUFraction: m.GCCPUFraction,
	}
}

// String returns a formatted string representation of memory stats.
func (m MemoryStats) String() string {
	return fmt.Sprintf("Alloc: %d KB, Total: %d KB, Sys: %d KB, GC: %d (%.2f%% CPU)",
		m.Alloc/1024,
		m.TotalAlloc/1024,
		m.Sys/1024,
		m.NumGC,
		m.GCCPUFraction*100)
}

// BenchmarkResult holds benchmark results.
type BenchmarkResult struct {
	Name         string
	Duration     time.Duration
	MemoryBefore MemoryStats
	MemoryAfter  MemoryStats
	Iterations   int
	Error        error
}

// String returns a formatted string representation of the benchmark result.
func (br BenchmarkResult) String() string {
	if br.Error != nil {
		return fmt.Sprintf("%s: ERROR - %v", br.Name, br.Error)
	}

	memDiff := br.MemoryAfter.Alloc - br.MemoryBefore.Alloc
	avgDuration := br.Duration / time.Duration(br.Iterations)

	return fmt.Sprintf("%s: %d iterations, avg: %v, total: %v, mem: +%d KB",
		br.Name, br.Iterations, avgDuration, br.Duration,
		int64(memDiff)/1024) //nolint:gosec // G115: Safe conversion for memory display
}

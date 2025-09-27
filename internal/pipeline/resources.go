package pipeline

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// ResourceManager manages memory and concurrency limits during parallel processing.
type ResourceManager struct {
	maxMemoryBytes   uint64
	maxGoroutines    int
	memoryThreshold  float64 // Threshold for memory pressure (0.0-1.0)
	goroutineSem     chan struct{}
	memoryMonitor    *MemoryMonitor
	stats            ResourceStats
	statsMutex       sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	monitoringActive bool
}

// ResourceStats holds resource usage statistics.
type ResourceStats struct {
	CurrentMemoryBytes   uint64        `json:"current_memory_bytes"`
	PeakMemoryBytes      uint64        `json:"peak_memory_bytes"`
	ActiveGoroutines     int           `json:"active_goroutines"`
	PeakGoroutines       int           `json:"peak_goroutines"`
	MemoryPressureEvents int           `json:"memory_pressure_events"`
	GoroutineBlocks      int           `json:"goroutine_blocks"`
	LastMemoryPressure   time.Time     `json:"last_memory_pressure"`
	MonitoringDuration   time.Duration `json:"monitoring_duration"`
	MemoryUtilization    float64       `json:"memory_utilization"`    // 0.0-1.0
	GoroutineUtilization float64       `json:"goroutine_utilization"` // 0.0-1.0
}

// ResourceConfig holds configuration for resource management.
type ResourceConfig struct {
	MaxMemoryBytes      uint64        // Maximum memory usage in bytes (0 = no limit)
	MaxGoroutines       int           // Maximum concurrent goroutines (0 = no limit)
	MemoryThreshold     float64       // Memory pressure threshold 0.0-1.0 (default: 0.8)
	MonitorInterval     time.Duration // How often to check resource usage
	EnableBackpressure  bool          // Whether to apply backpressure when resources are constrained
	EnableAdaptiveScale bool          // Whether to adaptively scale workers based on resource usage
}

// DefaultResourceConfig returns sensible defaults for resource management.
func DefaultResourceConfig() ResourceConfig {
	return ResourceConfig{
		MaxMemoryBytes:      0,   // No memory limit by default
		MaxGoroutines:       0,   // No goroutine limit by default
		MemoryThreshold:     0.8, // 80% memory usage triggers pressure
		MonitorInterval:     time.Second,
		EnableBackpressure:  true,
		EnableAdaptiveScale: false,
	}
}

// NewResourceManager creates a new resource manager with the given configuration.
func NewResourceManager(config ResourceConfig) *ResourceManager {
	ctx, cancel := context.WithCancel(context.Background())

	rm := &ResourceManager{
		maxMemoryBytes:   config.MaxMemoryBytes,
		maxGoroutines:    config.MaxGoroutines,
		memoryThreshold:  config.MemoryThreshold,
		memoryMonitor:    NewMemoryMonitor(config.MonitorInterval),
		ctx:              ctx,
		cancel:           cancel,
		monitoringActive: false,
	}

	// Initialize goroutine semaphore if limit is set
	if config.MaxGoroutines > 0 {
		rm.goroutineSem = make(chan struct{}, config.MaxGoroutines)
	}

	// Validate threshold
	if rm.memoryThreshold <= 0 || rm.memoryThreshold > 1.0 {
		rm.memoryThreshold = 0.8
	}

	return rm
}

// Start begins resource monitoring.
func (rm *ResourceManager) Start() {
	rm.statsMutex.Lock()
	defer rm.statsMutex.Unlock()

	if rm.monitoringActive {
		return
	}

	rm.monitoringActive = true
	rm.memoryMonitor.Start()

	// Start background monitoring
	go rm.monitorResources()
}

// Stop stops resource monitoring and releases resources.
func (rm *ResourceManager) Stop() {
	rm.cancel()

	rm.statsMutex.Lock()
	defer rm.statsMutex.Unlock()

	if rm.memoryMonitor != nil {
		rm.memoryMonitor.Stop()
	}
	rm.monitoringActive = false
}

// AcquireGoroutine attempts to acquire a goroutine slot.
// Returns an error if the limit is exceeded and context is cancelled.
func (rm *ResourceManager) AcquireGoroutine(ctx context.Context) error {
	if rm.goroutineSem == nil {
		// No limit
		rm.updateGoroutineStats(1)
		return nil
	}

	select {
	case rm.goroutineSem <- struct{}{}:
		rm.updateGoroutineStats(1)
		return nil
	case <-ctx.Done():
		rm.statsMutex.Lock()
		rm.stats.GoroutineBlocks++
		rm.statsMutex.Unlock()
		return ctx.Err()
	}
}

// ReleaseGoroutine releases a goroutine slot.
func (rm *ResourceManager) ReleaseGoroutine() {
	if rm.goroutineSem != nil {
		select {
		case <-rm.goroutineSem:
		default:
			// Should not happen, but don't block
		}
	}
	rm.updateGoroutineStats(-1)
}

// CheckMemoryPressure returns true if memory usage is above the threshold.
func (rm *ResourceManager) CheckMemoryPressure() bool {
	if rm.maxMemoryBytes == 0 {
		return false
	}

	current := rm.memoryMonitor.GetCurrentUsage()
	utilization := float64(current) / float64(rm.maxMemoryBytes)

	rm.statsMutex.Lock()
	defer rm.statsMutex.Unlock()

	rm.stats.MemoryUtilization = utilization

	if utilization > rm.memoryThreshold {
		rm.stats.MemoryPressureEvents++
		rm.stats.LastMemoryPressure = time.Now()
		return true
	}

	return false
}

// GetStats returns a copy of current resource statistics.
func (rm *ResourceManager) GetStats() ResourceStats {
	rm.statsMutex.RLock()
	defer rm.statsMutex.RUnlock()

	stats := rm.stats
	stats.CurrentMemoryBytes = rm.memoryMonitor.GetCurrentUsage()

	if rm.maxGoroutines > 0 {
		stats.GoroutineUtilization = float64(stats.ActiveGoroutines) / float64(rm.maxGoroutines)
	}

	return stats
}

// ShouldThrottle returns true if processing should be throttled due to resource constraints.
func (rm *ResourceManager) ShouldThrottle() bool {
	return rm.CheckMemoryPressure()
}

// GetOptimalWorkerCount returns the recommended number of workers based on current resource usage.
func (rm *ResourceManager) GetOptimalWorkerCount() int {
	// Start with CPU count as baseline
	optimal := runtime.NumCPU()

	// Apply goroutine limit if set
	if rm.maxGoroutines > 0 && optimal > rm.maxGoroutines {
		optimal = rm.maxGoroutines
	}

	// Reduce workers if under memory pressure
	if rm.CheckMemoryPressure() {
		optimal /= 2
		if optimal < 1 {
			optimal = 1
		}
	}

	return optimal
}

// updateGoroutineStats updates goroutine usage statistics.
func (rm *ResourceManager) updateGoroutineStats(delta int) {
	rm.statsMutex.Lock()
	defer rm.statsMutex.Unlock()

	rm.stats.ActiveGoroutines += delta
	if rm.stats.ActiveGoroutines < 0 {
		rm.stats.ActiveGoroutines = 0
	}

	if rm.stats.ActiveGoroutines > rm.stats.PeakGoroutines {
		rm.stats.PeakGoroutines = rm.stats.ActiveGoroutines
	}
}

// monitorResources runs in the background to collect resource statistics.
func (rm *ResourceManager) monitorResources() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	startTime := time.Now()

	for {
		select {
		case <-ticker.C:
			rm.updateMemoryStats()
			rm.statsMutex.Lock()
			rm.stats.MonitoringDuration = time.Since(startTime)
			rm.statsMutex.Unlock()

		case <-rm.ctx.Done():
			return
		}
	}
}

// updateMemoryStats updates memory usage statistics.
func (rm *ResourceManager) updateMemoryStats() {
	current := rm.memoryMonitor.GetCurrentUsage()

	rm.statsMutex.Lock()
	defer rm.statsMutex.Unlock()

	rm.stats.CurrentMemoryBytes = current
	if current > rm.stats.PeakMemoryBytes {
		rm.stats.PeakMemoryBytes = current
	}

	if rm.maxMemoryBytes > 0 {
		rm.stats.MemoryUtilization = float64(current) / float64(rm.maxMemoryBytes)
	}
}

// MemoryMonitor tracks memory usage over time.
type MemoryMonitor struct {
	interval     time.Duration
	currentUsage uint64
	peakUsage    uint64
	samples      []uint64
	maxSamples   int
	mutex        sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
	active       bool
}

// NewMemoryMonitor creates a new memory monitor.
func NewMemoryMonitor(interval time.Duration) *MemoryMonitor {
	// Ensure we have a positive interval for the ticker
	if interval <= 0 {
		interval = time.Second // Default to 1 second if not specified or invalid
	}

	ctx, cancel := context.WithCancel(context.Background())
	return &MemoryMonitor{
		interval:   interval,
		maxSamples: 60, // Keep last 60 samples
		samples:    make([]uint64, 0, 60),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start begins memory monitoring.
func (mm *MemoryMonitor) Start() {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	if mm.active {
		return
	}

	mm.active = true
	go mm.monitor()
}

// Stop stops memory monitoring.
func (mm *MemoryMonitor) Stop() {
	mm.cancel()

	mm.mutex.Lock()
	mm.active = false
	mm.mutex.Unlock()
}

// GetCurrentUsage returns the current memory usage in bytes.
func (mm *MemoryMonitor) GetCurrentUsage() uint64 {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.currentUsage
}

// GetPeakUsage returns the peak memory usage in bytes.
func (mm *MemoryMonitor) GetPeakUsage() uint64 {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()
	return mm.peakUsage
}

// GetAverageUsage returns the average memory usage over recent samples.
func (mm *MemoryMonitor) GetAverageUsage() uint64 {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()

	if len(mm.samples) == 0 {
		return mm.currentUsage
	}

	var sum uint64
	for _, sample := range mm.samples {
		sum += sample
	}
	return sum / uint64(len(mm.samples))
}

// monitor runs the memory monitoring loop.
func (mm *MemoryMonitor) monitor() {
	ticker := time.NewTicker(mm.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mm.updateUsage()
		case <-mm.ctx.Done():
			return
		}
	}
}

// updateUsage updates the current memory usage statistics.
func (mm *MemoryMonitor) updateUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	mm.currentUsage = m.Alloc
	if mm.currentUsage > mm.peakUsage {
		mm.peakUsage = mm.currentUsage
	}

	// Add to samples
	mm.samples = append(mm.samples, mm.currentUsage)
	if len(mm.samples) > mm.maxSamples {
		// Remove oldest sample
		copy(mm.samples, mm.samples[1:])
		mm.samples = mm.samples[:mm.maxSamples]
	}
}

// AdaptiveWorkerPool automatically adjusts the number of workers based on resource usage.
type AdaptiveWorkerPool struct {
	resourceManager *ResourceManager
	currentWorkers  int
	minWorkers      int
	maxWorkers      int
	adjustInterval  time.Duration
	mutex           sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	active          bool
}

// NewAdaptiveWorkerPool creates a new adaptive worker pool.
func NewAdaptiveWorkerPool(rm *ResourceManager, minWorkers, maxWorkers int, adjustInterval time.Duration) *AdaptiveWorkerPool {
	ctx, cancel := context.WithCancel(context.Background())

	if minWorkers <= 0 {
		minWorkers = 1
	}
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	if minWorkers > maxWorkers {
		minWorkers = maxWorkers
	}

	return &AdaptiveWorkerPool{
		resourceManager: rm,
		currentWorkers:  minWorkers,
		minWorkers:      minWorkers,
		maxWorkers:      maxWorkers,
		adjustInterval:  adjustInterval,
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Start begins adaptive scaling.
func (awp *AdaptiveWorkerPool) Start() {
	awp.mutex.Lock()
	defer awp.mutex.Unlock()

	if awp.active {
		return
	}

	awp.active = true
	go awp.scaleWorkers()
}

// Stop stops adaptive scaling.
func (awp *AdaptiveWorkerPool) Stop() {
	awp.cancel()

	awp.mutex.Lock()
	awp.active = false
	awp.mutex.Unlock()
}

// GetCurrentWorkerCount returns the current number of workers.
func (awp *AdaptiveWorkerPool) GetCurrentWorkerCount() int {
	awp.mutex.RLock()
	defer awp.mutex.RUnlock()
	return awp.currentWorkers
}

// scaleWorkers adjusts the worker count based on resource usage.
func (awp *AdaptiveWorkerPool) scaleWorkers() {
	ticker := time.NewTicker(awp.adjustInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			awp.adjustWorkerCount()
		case <-awp.ctx.Done():
			return
		}
	}
}

// adjustWorkerCount adjusts the number of workers based on current resource usage.
func (awp *AdaptiveWorkerPool) adjustWorkerCount() {
	optimal := awp.resourceManager.GetOptimalWorkerCount()

	// Clamp to min/max bounds
	if optimal < awp.minWorkers {
		optimal = awp.minWorkers
	}
	if optimal > awp.maxWorkers {
		optimal = awp.maxWorkers
	}

	awp.mutex.Lock()
	awp.currentWorkers = optimal
	awp.mutex.Unlock()
}

// ResourceError represents an error related to resource management.
type ResourceError struct {
	Type    string
	Message string
	Stats   ResourceStats
}

func (e ResourceError) Error() string {
	return fmt.Sprintf("resource error (%s): %s", e.Type, e.Message)
}

// NewMemoryLimitError creates a new memory limit error.
func NewMemoryLimitError(current, limit uint64) *ResourceError {
	return &ResourceError{
		Type:    "memory_limit",
		Message: fmt.Sprintf("memory usage %d bytes exceeds limit %d bytes", current, limit),
	}
}

// NewGoroutineLimitError creates a new goroutine limit error.
func NewGoroutineLimitError(current, limit int) *ResourceError {
	return &ResourceError{
		Type:    "goroutine_limit",
		Message: fmt.Sprintf("goroutine count %d exceeds limit %d", current, limit),
	}
}

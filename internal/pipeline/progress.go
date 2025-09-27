package pipeline

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"
)

// ProgressCallback defines the interface for progress reporting during batch processing.
type ProgressCallback interface {
	// OnStart is called when processing begins with the total number of items.
	OnStart(total int)

	// OnProgress is called periodically with current progress.
	OnProgress(current, total int)

	// OnComplete is called when processing is finished.
	OnComplete()

	// OnError is called when an error occurs (optional).
	OnError(current int, err error)
}

// NoOpProgressCallback implements ProgressCallback but does nothing.
// Useful as a default when no progress reporting is needed.
type NoOpProgressCallback struct{}

func (NoOpProgressCallback) OnStart(total int)              {}
func (NoOpProgressCallback) OnProgress(current, total int)  {}
func (NoOpProgressCallback) OnComplete()                    {}
func (NoOpProgressCallback) OnError(current int, err error) {}

// ConsoleProgressCallback displays a progress bar on the console.
type ConsoleProgressCallback struct {
	writer         io.Writer
	prefix         string
	width          int
	lastUpdate     time.Time
	updateInterval time.Duration
	mutex          sync.Mutex
	startTime      time.Time
	showETA        bool
	showRate       bool
}

// NewConsoleProgressCallback creates a new console progress reporter.
func NewConsoleProgressCallback(writer io.Writer, prefix string) *ConsoleProgressCallback {
	if writer == nil {
		writer = os.Stderr
	}
	return &ConsoleProgressCallback{
		writer:         writer,
		prefix:         prefix,
		width:          50,
		updateInterval: 100 * time.Millisecond,
		showETA:        true,
		showRate:       true,
	}
}

// WithWidth sets the progress bar width.
func (c *ConsoleProgressCallback) WithWidth(width int) *ConsoleProgressCallback {
	c.width = width
	return c
}

// WithUpdateInterval sets how frequently the progress bar updates.
func (c *ConsoleProgressCallback) WithUpdateInterval(interval time.Duration) *ConsoleProgressCallback {
	c.updateInterval = interval
	return c
}

// WithOptions configures display options.
func (c *ConsoleProgressCallback) WithOptions(showETA, showRate bool) *ConsoleProgressCallback {
	c.showETA = showETA
	c.showRate = showRate
	return c
}

func (c *ConsoleProgressCallback) OnStart(total int) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.startTime = time.Now()
	c.lastUpdate = time.Time{}

	_, _ = fmt.Fprintf(c.writer, "%s0/%d (0.0%%)\n", c.prefix, total)
}

func (c *ConsoleProgressCallback) OnProgress(current, total int) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	now := time.Now()
	if now.Sub(c.lastUpdate) < c.updateInterval && current < total {
		return // Don't update too frequently
	}
	c.lastUpdate = now

	c.drawProgressBar(current, total, now)
}

func (c *ConsoleProgressCallback) OnComplete() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	elapsed := time.Since(c.startTime)
	_, _ = fmt.Fprintf(c.writer, "\n%sCompleted in %v\n", c.prefix, elapsed.Round(time.Millisecond))
}

func (c *ConsoleProgressCallback) OnError(current int, err error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	_, _ = fmt.Fprintf(c.writer, "\n%sError at item %d: %v\n", c.prefix, current, err)
}

func (c *ConsoleProgressCallback) drawProgressBar(current, total int, now time.Time) {
	if total == 0 {
		return
	}

	percent := float64(current) / float64(total) * 100.0
	filled := int(float64(c.width) * float64(current) / float64(total))

	bar := strings.Repeat("█", filled) + strings.Repeat("░", c.width-filled)

	// Build status line
	status := fmt.Sprintf("\r%s[%s] %d/%d (%.1f%%)", c.prefix, bar, current, total, percent)

	// Add rate and ETA if enabled
	if !c.showRate && !c.showETA {
		_, _ = fmt.Fprint(c.writer, status)
		return
	}
	elapsed := now.Sub(c.startTime)
	if elapsed <= 0 || current <= 0 {
		_, _ = fmt.Fprint(c.writer, status)
		return
	}
	if c.showRate {
		rate := float64(current) / elapsed.Seconds()
		status += fmt.Sprintf(" %.1f/s", rate)
	}
	if c.showETA && current < total {
		remaining := total - current
		etaSeconds := elapsed.Seconds() * float64(remaining) / float64(current)
		eta := time.Duration(etaSeconds) * time.Second
		status += fmt.Sprintf(" ETA: %v", eta.Round(time.Second))
	}

	_, _ = fmt.Fprint(c.writer, status)
}

// LogProgressCallback logs progress updates using slog.
type LogProgressCallback struct {
	logger    *slog.Logger
	level     slog.Level
	prefix    string
	interval  int // Log every N items
	lastLog   int
	startTime time.Time
}

// NewLogProgressCallback creates a new log-based progress reporter.
func NewLogProgressCallback(logger *slog.Logger, level slog.Level, prefix string) *LogProgressCallback {
	if logger == nil {
		logger = slog.Default()
	}
	return &LogProgressCallback{
		logger:   logger,
		level:    level,
		prefix:   prefix,
		interval: 10, // Log every 10 items by default
	}
}

// WithInterval sets how frequently to log progress (every N items).
func (l *LogProgressCallback) WithInterval(interval int) *LogProgressCallback {
	l.interval = interval
	return l
}

func (l *LogProgressCallback) OnStart(total int) {
	l.startTime = time.Now()
	l.lastLog = 0
	l.logger.Log(nil, l.level, l.prefix+"Starting processing", "total", total)
}

func (l *LogProgressCallback) OnProgress(current, total int) {
	if current-l.lastLog >= l.interval || current == total {
		l.lastLog = current
		percent := float64(current) / float64(total) * 100.0
		elapsed := time.Since(l.startTime)
		rate := float64(current) / elapsed.Seconds()

		l.logger.Log(nil, l.level, l.prefix+"Progress update",
			"current", current,
			"total", total,
			"percent", fmt.Sprintf("%.1f", percent),
			"rate", fmt.Sprintf("%.1f/s", rate),
			"elapsed", elapsed.Round(time.Millisecond),
		)
	}
}

func (l *LogProgressCallback) OnComplete() {
	elapsed := time.Since(l.startTime)
	l.logger.Log(nil, l.level, l.prefix+"Processing completed", "elapsed", elapsed.Round(time.Millisecond))
}

func (l *LogProgressCallback) OnError(current int, err error) {
	l.logger.Log(nil, slog.LevelError, l.prefix+"Processing error", "current", current, "error", err)
}

// MultiProgressCallback combines multiple progress callbacks.
type MultiProgressCallback struct {
	callbacks []ProgressCallback
}

// NewMultiProgressCallback creates a progress callback that reports to multiple callbacks.
func NewMultiProgressCallback(callbacks ...ProgressCallback) *MultiProgressCallback {
	return &MultiProgressCallback{callbacks: callbacks}
}

// Add adds another progress callback.
func (m *MultiProgressCallback) Add(callback ProgressCallback) {
	m.callbacks = append(m.callbacks, callback)
}

func (m *MultiProgressCallback) OnStart(total int) {
	for _, cb := range m.callbacks {
		cb.OnStart(total)
	}
}

func (m *MultiProgressCallback) OnProgress(current, total int) {
	for _, cb := range m.callbacks {
		cb.OnProgress(current, total)
	}
}

func (m *MultiProgressCallback) OnComplete() {
	for _, cb := range m.callbacks {
		cb.OnComplete()
	}
}

func (m *MultiProgressCallback) OnError(current int, err error) {
	for _, cb := range m.callbacks {
		cb.OnError(current, err)
	}
}

// ThrottledProgressCallback wraps another callback and throttles updates.
type ThrottledProgressCallback struct {
	wrapped     ProgressCallback
	minInterval time.Duration
	lastUpdate  time.Time
	mutex       sync.Mutex
}

// NewThrottledProgressCallback creates a throttled wrapper around another callback.
func NewThrottledProgressCallback(wrapped ProgressCallback, minInterval time.Duration) *ThrottledProgressCallback {
	return &ThrottledProgressCallback{
		wrapped:     wrapped,
		minInterval: minInterval,
	}
}

func (t *ThrottledProgressCallback) OnStart(total int) {
	t.wrapped.OnStart(total)
}

func (t *ThrottledProgressCallback) OnProgress(current, total int) {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	now := time.Now()
	if current == total || t.lastUpdate.IsZero() || now.Sub(t.lastUpdate) >= t.minInterval {
		t.lastUpdate = now
		t.wrapped.OnProgress(current, total)
	}
}

func (t *ThrottledProgressCallback) OnComplete() {
	t.wrapped.OnComplete()
}

func (t *ThrottledProgressCallback) OnError(current int, err error) {
	t.wrapped.OnError(current, err)
}

// ProgressTracker tracks detailed progress statistics.
type ProgressTracker struct {
	StartTime      time.Time     `json:"start_time"`
	Total          int           `json:"total"`
	Current        int           `json:"current"`
	Completed      int           `json:"completed"`
	Failed         int           `json:"failed"`
	Rate           float64       `json:"rate_per_second"`
	EstimatedTotal time.Duration `json:"estimated_total_duration"`
	Elapsed        time.Duration `json:"elapsed_duration"`
	mutex          sync.RWMutex
}

// NewProgressTracker creates a new progress tracker.
func NewProgressTracker(total int) *ProgressTracker {
	return &ProgressTracker{
		StartTime: time.Now(),
		Total:     total,
	}
}

// Update updates the progress tracker.
func (pt *ProgressTracker) Update(current, completed, failed int) {
	pt.mutex.Lock()
	defer pt.mutex.Unlock()

	pt.Current = current
	pt.Completed = completed
	pt.Failed = failed
	pt.Elapsed = time.Since(pt.StartTime)

	if pt.Elapsed > 0 && pt.Current > 0 {
		pt.Rate = float64(pt.Current) / pt.Elapsed.Seconds()
		if pt.Rate > 0 {
			remaining := pt.Total - pt.Current
			etaSeconds := float64(remaining) / pt.Rate
			pt.EstimatedTotal = time.Duration(etaSeconds * float64(time.Second))
		}
	}
}

// GetStats returns a copy of current statistics.
func (pt *ProgressTracker) GetStats() ProgressTracker {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()

	return ProgressTracker{
		StartTime:      pt.StartTime,
		Total:          pt.Total,
		Current:        pt.Current,
		Completed:      pt.Completed,
		Failed:         pt.Failed,
		Rate:           pt.Rate,
		EstimatedTotal: pt.EstimatedTotal,
		Elapsed:        pt.Elapsed,
	}
}

// PercentComplete returns the completion percentage.
func (pt *ProgressTracker) PercentComplete() float64 {
	pt.mutex.RLock()
	defer pt.mutex.RUnlock()

	if pt.Total == 0 {
		return 0
	}
	return float64(pt.Current) / float64(pt.Total) * 100.0
}

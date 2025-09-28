// Package common provides shared utilities including timing functionality.
package common

import (
	"fmt"
	"time"
)

// Timer provides timing utilities for benchmarking with optional naming.
type Timer struct {
	start    time.Time
	name     string
	duration time.Duration
}

// NewTimer creates a new timer.
func NewTimer() *Timer {
	return &Timer{start: time.Now()}
}

// NewNamedTimer creates a new timer with the given name.
func NewNamedTimer(name string) *Timer {
	return &Timer{
		name:  name,
		start: time.Now(),
	}
}

// Stop stops the timer and returns the elapsed duration.
func (t *Timer) Stop() time.Duration {
	t.duration = time.Since(t.start)
	return t.duration
}

// Duration returns the recorded duration (only valid after Stop()).
func (t *Timer) Duration() time.Duration {
	return t.duration
}

// Name returns the timer name (empty string if unnamed).
func (t *Timer) Name() string {
	return t.name
}

// String returns a formatted string representation of the timer.
func (t *Timer) String() string {
	if t.name != "" {
		return fmt.Sprintf("%s: %v", t.name, t.duration)
	}
	return fmt.Sprintf("%v", t.duration)
}

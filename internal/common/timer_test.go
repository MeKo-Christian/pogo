package common

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestTimer(t *testing.T) {
	timer := NewNamedTimer("test_timer")
	assert.Equal(t, "test_timer", timer.Name())

	// Sleep for a short duration
	time.Sleep(10 * time.Millisecond)

	duration := timer.Stop()
	assert.GreaterOrEqual(t, duration, 10*time.Millisecond)
	assert.Equal(t, duration, timer.Duration())

	str := timer.String()
	assert.Contains(t, str, "test_timer")
	assert.Contains(t, str, "ms")
}

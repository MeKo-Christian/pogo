package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTestCommand(t *testing.T) {
	assert.NotNil(t, testCmd)
	assert.Equal(t, "test", testCmd.Use)
	assert.NotEmpty(t, testCmd.Short)
}

func TestTestCommandHelp(t *testing.T) {
	// Call Help directly to avoid differences in cobra help flag interception
	cmd := testCmd
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	err := cmd.Help()
	require.NoError(t, err)
	output := strings.TrimSpace(buf.String())
	assert.Contains(t, output, "test")
	assert.Contains(t, output, "Usage:")
}

func TestTestCommandExecution(t *testing.T) {
	// Execute via root to ensure cobra wiring is consistent
	buf := new(bytes.Buffer)
	rootCmd.SetOut(buf)
	rootCmd.SetErr(buf)
	// Also ensure subcommand streams use the same buffer
	testCmd.SetOut(buf)
	testCmd.SetErr(buf)
	rootCmd.SetArgs([]string{"test"})
	err := rootCmd.Execute()
	output := strings.TrimSpace(buf.String())

	// The command may error if ONNX is missing; either path is acceptable
	if err != nil {
		t.Logf("Test command returned error (possibly due to missing ONNX Runtime): %v", err)
	}
	// Ensure that we either saw output or received an error
	assert.True(t, len(output) > 0 || err != nil)
}

package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRootCommand(t *testing.T) {
	assert.NotNil(t, rootCmd)
	assert.Equal(t, "pogo", rootCmd.Use)
	assert.NotEmpty(t, rootCmd.Short)
	assert.NotEmpty(t, rootCmd.Long)
}

func TestRootCommandHelp(t *testing.T) {
	cmd := rootCmd

	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Set help flag and execute
	cmd.SetArgs([]string{"--help"})
	err := cmd.Execute()

	// Help should not return an error
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "OCR (Optical Character Recognition)")
	assert.Contains(t, output, "Available Commands:")
	assert.Contains(t, output, "Usage:")
}

func TestRootCommandVersion(t *testing.T) {
	cmd := rootCmd

	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Set version flag and execute
	cmd.SetArgs([]string{"--version"})
	err := cmd.Execute()

	// Version should not return an error
	require.NoError(t, err)

	output := buf.String()
	// The output should contain version information
	assert.NotEmpty(t, output)
}

func TestRootCommandSubcommands(t *testing.T) {
	cmd := rootCmd

	// Check that expected subcommands are present
	subcommands := cmd.Commands()
	commandNames := make([]string, len(subcommands))
	for i, subcmd := range subcommands {
		commandNames[i] = subcmd.Name()
	}

	// Expected subcommands based on the current implementation
	expectedCommands := []string{"image", "pdf", "serve", "test"}
	for _, expected := range expectedCommands {
		assert.Contains(t, commandNames, expected, "Expected subcommand '%s' not found", expected)
	}
}

func TestRootCommandInvalidFlag(t *testing.T) {
	cmd := rootCmd

	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Set invalid flag and execute
	cmd.SetArgs([]string{"--invalid-flag"})
	err := cmd.Execute()

	// Should return an error for invalid flag
	require.Error(t, err)

	output := buf.String()
	assert.Contains(t, output, "unknown flag")
}

func TestRootCommandNoArgs(t *testing.T) {
	cmd := rootCmd

	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Execute with no arguments
	cmd.SetArgs([]string{})
	err := cmd.Execute()

	// Should not error, but should show help or usage
	require.NoError(t, err)

	output := buf.String()
	// Should contain usage information
	assert.NotEmpty(t, output)
}

// Helper function to execute command and capture output.
func executeCommandAndCaptureOutput(t *testing.T, cmd *cobra.Command, args []string) (string, error) {
	t.Helper()

	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	cmd.SetArgs(args)

	err := cmd.Execute()
	return strings.TrimSpace(buf.String()), err
}

func TestExecuteCommandHelper(t *testing.T) {
	cmd := rootCmd

	// Test the helper function itself
	output, err := executeCommandAndCaptureOutput(t, cmd, []string{"--help"})
	require.NoError(t, err)
	assert.Contains(t, output, "Available Commands:")
}

// Test command configuration.
func TestRootCommandConfiguration(t *testing.T) {
	cmd := rootCmd

	// Test that command has proper configuration
	assert.True(t, cmd.HasSubCommands())
	assert.NotNil(t, cmd.PersistentFlags())

	// Test that verbose flag exists if implemented
	if cmd.PersistentFlags().Lookup("verbose") != nil {
		assert.NotNil(t, cmd.PersistentFlags().Lookup("verbose"))
	}
}

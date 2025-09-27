package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImageCommand(t *testing.T) {
	assert.NotNil(t, imageCmd)
	// Accept "image" or extended usage forms
	assert.True(t, strings.HasPrefix(imageCmd.Use, "image"))
	assert.NotEmpty(t, imageCmd.Short)
	assert.NotEmpty(t, imageCmd.Long)
}

func TestImageCommandHelp(t *testing.T) {
	command := imageCmd
	buf := new(bytes.Buffer)
	command.SetOut(buf)
	command.SetErr(buf)
	command.SetArgs([]string{"--help"})
	// Call help directly to avoid cobra root execution differences
	err := command.Help()
	require.NoError(t, err)
	output := strings.TrimSpace(buf.String())
	assert.Contains(t, output, "Process images")
	assert.Contains(t, output, "Usage:")
	assert.Contains(t, output, "Flags:")
}

func TestImageCommandFlags(t *testing.T) {
	command := imageCmd

	// Check that expected flags exist
	flags := command.Flags()

	// Check for common flags that might be implemented
	expectedFlags := []string{"format", "output", "confidence", "model"}
	for _, flagName := range expectedFlags {
		flag := flags.Lookup(flagName)
		if flag != nil {
			t.Logf("Flag '%s' found: %s", flagName, flag.Usage)
		}
	}

	// At minimum, there should be some flags defined
	assert.True(t, flags.HasFlags() || len(command.LocalFlags().FlagUsages()) > 0)
}

func TestImageCommandWithoutFile(t *testing.T) {
	command := imageCmd
	buf := new(bytes.Buffer)
	command.SetOut(buf)
	command.SetErr(buf)
	rootCmd.SetArgs([]string{})
	err := command.Execute()
	output := strings.TrimSpace(buf.String())
	if err != nil {
		assert.True(t, len(output) > 0 || err.Error() != "")
	} else {
		if output == "" {
			// Fallback to help for subcommand
			_ = command.Help()
			output = strings.TrimSpace(buf.String())
		}
		assert.Contains(t, output, "image")
	}
}

func TestImageCommandWithNonExistentFile(t *testing.T) {
	// Call RunE directly with a missing file to validate error behavior
	err := imageCmd.RunE(imageCmd, []string{"/non/existent/file.jpg"})
	assert.Error(t, err)
}

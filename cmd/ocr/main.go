package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var (
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "go-oar-ocr",
		Short: "OCR pipeline for text detection and recognition",
		Long: `A Go implementation of the OAR-OCR pipeline providing text detection
and recognition capabilities with support for images and PDFs.`,
		Version: fmt.Sprintf("%s (commit: %s, built: %s)", version, commit, date),
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
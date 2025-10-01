package batch

import (
	"fmt"
	"os"
	"path/filepath"
)

// discoverImageFiles finds all image files matching the given patterns.
func discoverImageFiles(args []string, recursive bool, includePatterns, excludePatterns []string) ([]string, error) {
	var imageFiles []string

	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			return nil, fmt.Errorf("cannot access %s: %w", arg, err)
		}

		if info.IsDir() {
			files, err := discoverInDirectory(arg, recursive, includePatterns, excludePatterns)
			if err != nil {
				return nil, err
			}
			imageFiles = append(imageFiles, files...)
		} else if shouldIncludeFile(arg, includePatterns, excludePatterns) {
			imageFiles = append(imageFiles, arg)
		}
	}

	return imageFiles, nil
}

// discoverInDirectory recursively discovers image files in a directory.
func discoverInDirectory(dir string, recursive bool, includePatterns, excludePatterns []string) ([]string, error) {
	var files []string

	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if !recursive && path != dir {
				return filepath.SkipDir
			}
			return nil
		}

		if shouldIncludeFile(path, includePatterns, excludePatterns) {
			files = append(files, path)
		}

		return nil
	}

	return files, filepath.Walk(dir, walkFn)
}

// shouldIncludeFile determines if a file should be included based on include/exclude patterns.
func shouldIncludeFile(path string, includePatterns, excludePatterns []string) bool {
	// Check exclude patterns first
	if matchesAnyPattern(path, excludePatterns) {
		return false
	}

	// If no include patterns, include all (that aren't excluded)
	if len(includePatterns) == 0 {
		return true
	}

	// Otherwise, must match at least one include pattern
	return matchesAnyPattern(path, includePatterns)
}

// matchesAnyPattern checks if a file path matches any of the given patterns.
func matchesAnyPattern(path string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}

	base := filepath.Base(path)
	for _, pattern := range patterns {
		if matched, _ := filepath.Match(pattern, base); matched {
			return true
		}
	}
	return false
}

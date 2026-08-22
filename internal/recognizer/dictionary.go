package recognizer

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
)

// Charset represents a recognition character set loaded from a dictionary file.
// Tokens can be single Unicode characters or multi-codepoint strings.
type Charset struct {
	Tokens       []string
	IndexToToken map[int]string
	TokenToIndex map[string]int
}

// removeBOM removes UTF-8 BOM if present from the first line.
func removeBOM(line string, isFirstLine bool) string {
	if isFirstLine {
		return strings.TrimPrefix(line, "\uFEFF")
	}
	return line
}

// processLine processes a single line from the dictionary file.
func processLine(line string, lineNum int) string {
	// Remove trailing newline but preserve other whitespace (spaces, tabs, ideographic spaces, etc.)
	line = strings.TrimSuffix(line, "\n")
	line = strings.TrimSuffix(line, "\r")
	line = removeBOM(line, lineNum == 1)
	return line
}

// buildCharsetMaps builds the index mappings from tokens.
func buildCharsetMaps(tokens []string) (map[int]string, map[string]int) {
	idxTo := make(map[int]string, len(tokens))
	toIdx := make(map[string]int, len(tokens))
	for i, t := range tokens {
		// If duplicates occur, keep the first occurrence
		if _, ok := toIdx[t]; !ok {
			toIdx[t] = i
		}
		idxTo[i] = t
	}
	return idxTo, toIdx
}

// SpaceToken is the token used for the trailing "space" class that PaddleOCR
// recognition models append after the dictionary entries. The full class layout
// of a PP-OCR CTC head is ["blank"] + dictionary + [SpaceToken].
const SpaceToken = " "

// CharsetOptions controls optional post-processing applied while a charset is built.
type CharsetOptions struct {
	// AppendSpace appends SpaceToken as the last token unless the dictionary
	// already contains it. PaddleOCR recognition models declare one more output
	// class than their dictionary has lines, and that extra class is the space
	// character. Defaults to false so the plain loaders keep their behaviour.
	AppendSpace bool
}

// readCharsetTokens reads a dictionary file and returns its tokens in file order.
func readCharsetTokens(path string) ([]string, error) {
	if path == "" {
		return nil, errors.New("dictionary path cannot be empty")
	}
	f, err := os.Open(path) //nolint:gosec // G304: Opening user-provided dictionary file is expected
	if err != nil {
		return nil, fmt.Errorf("failed to open dictionary: %w", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing dictionary file: %v\n", err)
		}
	}()

	scanner := bufio.NewScanner(f)
	tokens := make([]string, 0, 512)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := processLine(scanner.Text(), lineNum)
		// Skip truly empty lines (after removing BOM and newlines)
		// Whitespace-only lines should be preserved as they might be valid tokens
		if line == "" {
			continue
		}
		tokens = append(tokens, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed reading dictionary: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("dictionary is empty: %s", path)
	}
	return tokens, nil
}

// applyCharsetOptions applies the optional token adjustments. This is the single
// place where the space token is appended, so the single- and multi-file loaders
// cannot drift apart.
func applyCharsetOptions(tokens []string, opts CharsetOptions) []string {
	if !opts.AppendSpace {
		return tokens
	}
	for _, t := range tokens {
		if t == SpaceToken {
			return tokens
		}
	}
	return append(tokens, SpaceToken)
}

// newCharset finalises a token list into a Charset, applying opts first.
func newCharset(tokens []string, opts CharsetOptions) *Charset {
	tokens = applyCharsetOptions(tokens, opts)
	idxTo, toIdx := buildCharsetMaps(tokens)
	return &Charset{
		Tokens:       tokens,
		IndexToToken: idxTo,
		TokenToIndex: toIdx,
	}
}

// LoadCharset loads a dictionary file where each non-empty line is a token.
// Only the line terminator (\r\n or \n) and a leading UTF-8 BOM are stripped;
// all other whitespace is preserved, so a whitespace-only line stays a valid
// token. Lines that are empty after that stripping are skipped.
func LoadCharset(path string) (*Charset, error) {
	return LoadCharsetWithOptions(path, CharsetOptions{})
}

// LoadCharsetWithOptions loads a single dictionary file with the given options.
func LoadCharsetWithOptions(path string, opts CharsetOptions) (*Charset, error) {
	tokens, err := readCharsetTokens(path)
	if err != nil {
		return nil, err
	}
	return newCharset(tokens, opts), nil
}

// LoadCharsets merges multiple dictionary files into a single Charset.
// Tokens are appended in file order with de-duplication (first occurrence wins).
func LoadCharsets(paths []string) (*Charset, error) {
	return LoadCharsetsWithOptions(paths, CharsetOptions{})
}

// LoadCharsetsWithOptions merges multiple dictionary files with the given options.
func LoadCharsetsWithOptions(paths []string, opts CharsetOptions) (*Charset, error) {
	if len(paths) == 0 {
		return nil, errors.New("no dictionary paths provided")
	}
	// Use a set to deduplicate
	seen := make(map[string]struct{}, 1024)
	tokens := make([]string, 0, 1024)
	for _, p := range paths {
		if p == "" {
			continue
		}
		fileTokens, err := readCharsetTokens(p)
		if err != nil {
			return nil, err
		}
		for _, t := range fileTokens {
			if _, ok := seen[t]; ok {
				continue
			}
			seen[t] = struct{}{}
			tokens = append(tokens, t)
		}
	}
	if len(tokens) == 0 {
		return nil, errors.New("merged dictionary is empty")
	}
	return newCharset(tokens, opts), nil
}

// Size returns the number of tokens in the charset.
func (c *Charset) Size() int { return len(c.Tokens) }

// LookupIndex returns the index of a token, or -1 if not present.
func (c *Charset) LookupIndex(token string) int {
	if c == nil {
		return -1
	}
	if idx, ok := c.TokenToIndex[token]; ok {
		return idx
	}
	return -1
}

// LookupToken returns the token for an index, or empty string if missing.
func (c *Charset) LookupToken(index int) string {
	if c == nil {
		return ""
	}
	if t, ok := c.IndexToToken[index]; ok {
		return t
	}
	return ""
}

// Contains checks if a token exists in the charset.
func (c *Charset) Contains(token string) bool {
	if c == nil {
		return false
	}
	_, ok := c.TokenToIndex[token]
	return ok
}

// Filter removes any runes/characters from the text that are not in this charset.
// Returns the filtered text containing only characters present in the charset.
// If the charset is nil, returns the original text unchanged.
func (c *Charset) Filter(text string) string {
	if c == nil || len(c.TokenToIndex) == 0 {
		return text
	}

	// Convert text to runes for proper Unicode handling
	runes := []rune(text)
	filtered := make([]rune, 0, len(runes))

	for _, r := range runes {
		// Check if this rune (as a string) is in our charset
		token := string(r)
		if c.Contains(token) {
			filtered = append(filtered, r)
		}
	}

	return string(filtered)
}

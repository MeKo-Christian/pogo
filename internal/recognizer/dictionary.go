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
	line = strings.TrimSpace(line)
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

// LoadCharset loads a dictionary file where each non-empty line is a token.
// Leading/trailing whitespace is trimmed. UTF-8 BOM is removed if present.
func LoadCharset(path string) (*Charset, error) {
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

	idxTo, toIdx := buildCharsetMaps(tokens)

	return &Charset{
		Tokens:       tokens,
		IndexToToken: idxTo,
		TokenToIndex: toIdx,
	}, nil
}

// LoadCharsets merges multiple dictionary files into a single Charset.
// Tokens are appended in file order with de-duplication (first occurrence wins).
func LoadCharsets(paths []string) (*Charset, error) {
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
		cs, err := LoadCharset(p)
		if err != nil {
			return nil, err
		}
		for _, t := range cs.Tokens {
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
	idxTo := make(map[int]string, len(tokens))
	toIdx := make(map[string]int, len(tokens))
	for i, t := range tokens {
		if _, ok := toIdx[t]; !ok {
			toIdx[t] = i
		}
		idxTo[i] = t
	}
	return &Charset{Tokens: tokens, IndexToToken: idxTo, TokenToIndex: toIdx}, nil
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

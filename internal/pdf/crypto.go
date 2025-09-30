package pdf

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/pdfcpu/pdfcpu/pkg/api"
	"github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// PasswordCredentials contains the passwords for a PDF file.
type PasswordCredentials struct {
	UserPassword  string `json:"user_password,omitempty"`
	OwnerPassword string `json:"owner_password,omitempty"`
}

// PasswordHandler handles password-related operations for PDF files.
type PasswordHandler struct {
	allowPasswordPrompt bool
	defaultCredentials  *PasswordCredentials
}

// NewPasswordHandler creates a new password handler.
func NewPasswordHandler(allowPrompt bool) *PasswordHandler {
	return &PasswordHandler{
		allowPasswordPrompt: allowPrompt,
	}
}

// SetDefaultCredentials sets default credentials to try before prompting.
func (h *PasswordHandler) SetDefaultCredentials(creds *PasswordCredentials) {
	h.defaultCredentials = creds
}

// IsEncrypted checks if a PDF file is encrypted/password-protected.
func (h *PasswordHandler) IsEncrypted(filename string) (bool, error) {
	// Try to get page count - this will fail if the file is encrypted and we don't have the password
	_, err := api.PageCountFile(filename)
	if err != nil {
		// Check if the error indicates encryption
		if strings.Contains(err.Error(), "encrypted") ||
			strings.Contains(err.Error(), "password") ||
			strings.Contains(err.Error(), "decrypt") {
			return true, nil
		}
		// Some other error occurred
		return false, fmt.Errorf("failed to check PDF encryption status: %w", err)
	}

	// File is not encrypted
	return false, nil
}

// DecryptPDF decrypts a password-protected PDF file and returns the path to the decrypted temporary file.
// The caller is responsible for cleaning up the temporary file.
func (h *PasswordHandler) DecryptPDF(filename string, creds *PasswordCredentials) (string, error) {
	// Check if file is actually encrypted
	encrypted, err := h.IsEncrypted(filename)
	if err != nil {
		return "", err
	}

	if !encrypted {
		return filename, nil
	}

	config := h.createDecryptionConfig(creds)
	tempFileName, err := h.createTempFile()
	if err != nil {
		return "", err
	}

	// Try to decrypt with current credentials
	if err := h.tryDecryptWithConfig(filename, tempFileName, config); err == nil {
		return tempFileName, nil
	}

	// If decryption failed and we allow prompting, try to get password from user
	if h.allowPasswordPrompt {
		if tempFileName, err := h.tryDecryptWithPrompt(filename, tempFileName, config); err == nil {
			return tempFileName, nil
		}
	}

	// Clean up temp file if decryption failed
	_ = os.Remove(tempFileName)
	return "", fmt.Errorf("failed to decrypt PDF: %w", err)
}

// createDecryptionConfig creates a configuration with the provided credentials.
func (h *PasswordHandler) createDecryptionConfig(creds *PasswordCredentials) *model.Configuration {
	config := model.NewDefaultConfiguration()

	if creds != nil {
		if creds.UserPassword != "" {
			config.UserPW = creds.UserPassword
		}
		if creds.OwnerPassword != "" {
			config.OwnerPW = creds.OwnerPassword
		}
	} else if h.defaultCredentials != nil {
		config.UserPW = h.defaultCredentials.UserPassword
		config.OwnerPW = h.defaultCredentials.OwnerPassword
	}

	return config
}

// createTempFile creates a temporary file for decrypted PDF.
func (h *PasswordHandler) createTempFile() (string, error) {
	tempFile, err := os.CreateTemp("", "decrypted-*.pdf")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary file: %w", err)
	}
	_ = tempFile.Close() // Close file handle, keep path
	return tempFile.Name(), nil
}

// tryDecryptWithPrompt attempts to decrypt using prompted passwords.
func (h *PasswordHandler) tryDecryptWithPrompt(filename, tempFileName string,
	config *model.Configuration) (string, error) {
	promptCreds, promptErr := h.promptForPasswords()
	if promptErr != nil || promptCreds == nil {
		return "", fmt.Errorf("password prompting failed: %w", promptErr)
	}

	// Update config with prompted passwords
	if promptCreds.UserPassword != "" {
		config.UserPW = promptCreds.UserPassword
	}
	if promptCreds.OwnerPassword != "" {
		config.OwnerPW = promptCreds.OwnerPassword
	}

	// Try decryption again with prompted passwords
	err := h.tryDecryptWithConfig(filename, tempFileName, config)
	if err != nil {
		return "", err
	}

	return tempFileName, nil
}

// tryDecryptWithConfig attempts to decrypt a PDF with the given configuration.
func (h *PasswordHandler) tryDecryptWithConfig(inFile, outFile string, config *model.Configuration) error {
	return api.DecryptFile(inFile, outFile, config)
}

// promptForPasswords prompts the user for PDF passwords.
func (h *PasswordHandler) promptForPasswords() (*PasswordCredentials, error) {
	creds := &PasswordCredentials{}

	// Prompt for user password
	fmt.Print("Enter user password (or press Enter to skip): ")
	userPW, err := h.readPassword()
	if err != nil {
		return nil, fmt.Errorf("failed to read user password: %w", err)
	}
	creds.UserPassword = userPW

	// If no user password provided, prompt for owner password
	if userPW == "" {
		fmt.Print("Enter owner password (or press Enter to skip): ")
		ownerPW, err := h.readPassword()
		if err != nil {
			return nil, fmt.Errorf("failed to read owner password: %w", err)
		}
		creds.OwnerPassword = ownerPW
	}

	// Return nil if no passwords were provided
	if creds.UserPassword == "" && creds.OwnerPassword == "" {
		return nil, errors.New("no passwords provided")
	}

	return creds, nil
}

// readPassword reads a password from stdin, with masking if possible.
func (h *PasswordHandler) readPassword() (string, error) {
	// Try to read password with masking (Unix-like systems)
	if password, err := h.readPasswordMasked(); err == nil {
		return password, nil
	}

	// Fallback to regular input reading
	fmt.Println("(Warning: password will be visible)")
	return h.readPasswordVisible()
}

// handlePasswordChar processes a single character for password input.
func handlePasswordChar(char byte, password *strings.Builder) {
	switch {
	case char == '\n' || char == '\r':
		fmt.Println() // New line
	case char == '\b' || char == 127: // Backspace or DEL
		if password.Len() > 0 {
			// Convert to string, remove last character, and recreate builder
			currentPassword := password.String()
			if len(currentPassword) > 0 {
				password.Reset()
				password.WriteString(currentPassword[:len(currentPassword)-1])
				fmt.Print("\b \b") // Erase character
			}
		}
	case char >= 32 && char <= 126: // Printable characters
		password.WriteByte(char)
		fmt.Print("*")
	}
}

// readPasswordMasked reads a password with character masking.
func (h *PasswordHandler) readPasswordMasked() (string, error) {
	// This is a simplified implementation
	// In a production system, you might want to use a library like golang.org/x/term
	// for proper password masking across platforms

	fmt.Print("Password: ")

	// Read character by character and mask with *
	var password strings.Builder
	for {
		var b [1]byte
		_, err := os.Stdin.Read(b[:])
		if err != nil {
			return "", err
		}

		char := b[0]
		handlePasswordChar(char, &password)

		// Check if we should break (newline characters)
		if char == '\n' || char == '\r' {
			break
		}
	}

	return password.String(), nil
}

// readPasswordVisible reads a password without masking (fallback).
func (h *PasswordHandler) readPasswordVisible() (string, error) {
	reader := bufio.NewReader(os.Stdin)
	password, err := reader.ReadString('\n')
	if err != nil {
		return "", err
	}

	// Remove trailing newline
	return strings.TrimSpace(password), nil
}

// ValidateCredentials validates that the provided credentials can decrypt the PDF.
func (h *PasswordHandler) ValidateCredentials(filename string, creds *PasswordCredentials) error {
	if creds == nil {
		return errors.New("no credentials provided")
	}

	config := model.NewDefaultConfiguration()
	config.UserPW = creds.UserPassword
	config.OwnerPW = creds.OwnerPassword

	// Create a temporary file to test decryption
	tempFile, err := os.CreateTemp("", "validate-*.pdf")
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %w", err)
	}
	defer func() { _ = os.Remove(tempFile.Name()) }()
	_ = tempFile.Close()

	// Try to decrypt
	err = h.tryDecryptWithConfig(filename, tempFile.Name(), config)
	if err != nil {
		return fmt.Errorf("invalid credentials: %w", err)
	}

	return nil
}

// CleanupTempFile removes a temporary decrypted file.
func (h *PasswordHandler) CleanupTempFile(filename string) error {
	if filename == "" {
		return nil
	}

	// Only remove files that look like our temp files
	if strings.Contains(filename, "decrypted-") && strings.HasSuffix(filename, ".pdf") {
		return os.Remove(filename)
	}

	return nil
}

// SecureString represents a string that should be handled securely.
type SecureString struct {
	value []byte
}

// NewSecureString creates a new secure string.
func NewSecureString(s string) *SecureString {
	return &SecureString{
		value: []byte(s),
	}
}

// String returns the string value.
func (s *SecureString) String() string {
	if s == nil || s.value == nil {
		return ""
	}
	return string(s.value)
}

// Clear securely clears the string from memory.
func (s *SecureString) Clear() {
	if s == nil || s.value == nil {
		return
	}

	// Overwrite memory with zeros
	for i := range s.value {
		s.value[i] = 0
	}
	s.value = nil
}

// GetPasswordPrompt returns a formatted password prompt.
func GetPasswordPrompt(filename string) string {
	caser := cases.Title(language.English)
	return fmt.Sprintf("The PDF file %q is password protected. %s",
		filename,
		caser.String("please provide the password"))
}

// IsPasswordError checks if an error is related to password/encryption issues.
func IsPasswordError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	passwordKeywords := []string{
		"password",
		"encrypted",
		"decrypt",
		"authentication",
		"unauthorized",
		"invalid credentials",
	}

	for _, keyword := range passwordKeywords {
		if strings.Contains(errStr, keyword) {
			return true
		}
	}

	return false
}

// init ensures that the terminal is in the correct mode for password reading.
func init() {
	// Set up terminal for password input (Unix-like systems)
	// This is a placeholder - in a real implementation you might want to
	// configure the terminal appropriately for secure password input
}

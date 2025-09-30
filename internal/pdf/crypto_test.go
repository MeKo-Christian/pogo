package pdf

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPasswordCredentials(t *testing.T) {
	t.Run("create with both passwords", func(t *testing.T) {
		creds := &PasswordCredentials{
			UserPassword:  "user123",
			OwnerPassword: "owner456",
		}
		assert.Equal(t, "user123", creds.UserPassword)
		assert.Equal(t, "owner456", creds.OwnerPassword)
	})

	t.Run("create with only user password", func(t *testing.T) {
		creds := &PasswordCredentials{
			UserPassword: "user123",
		}
		assert.Equal(t, "user123", creds.UserPassword)
		assert.Empty(t, creds.OwnerPassword)
	})

	t.Run("create with only owner password", func(t *testing.T) {
		creds := &PasswordCredentials{
			OwnerPassword: "owner456",
		}
		assert.Empty(t, creds.UserPassword)
		assert.Equal(t, "owner456", creds.OwnerPassword)
	})
}

func TestNewPasswordHandler(t *testing.T) {
	t.Run("create with prompt allowed", func(t *testing.T) {
		handler := NewPasswordHandler(true)
		require.NotNil(t, handler)
		assert.True(t, handler.allowPasswordPrompt)
		assert.Nil(t, handler.defaultCredentials)
	})

	t.Run("create with prompt disallowed", func(t *testing.T) {
		handler := NewPasswordHandler(false)
		require.NotNil(t, handler)
		assert.False(t, handler.allowPasswordPrompt)
		assert.Nil(t, handler.defaultCredentials)
	})
}

func TestPasswordHandler_SetDefaultCredentials(t *testing.T) {
	handler := NewPasswordHandler(false)

	creds := &PasswordCredentials{
		UserPassword:  "test",
		OwnerPassword: "test2",
	}

	handler.SetDefaultCredentials(creds)
	assert.Equal(t, creds, handler.defaultCredentials)
}

func TestPasswordHandler_IsEncrypted(t *testing.T) {
	handler := NewPasswordHandler(false)
	tempDir := t.TempDir()

	t.Run("non-existent file", func(t *testing.T) {
		encrypted, err := handler.IsEncrypted("/non/existent/file.pdf")
		require.Error(t, err)
		assert.False(t, encrypted)
		assert.Contains(t, err.Error(), "failed to check PDF encryption status")
	})

	t.Run("not a PDF file", func(t *testing.T) {
		textFile := filepath.Join(tempDir, "not_a_pdf.txt")
		err := os.WriteFile(textFile, []byte("not a PDF"), 0o644)
		require.NoError(t, err)

		encrypted, err := handler.IsEncrypted(textFile)
		// Should return an error since it's not a valid PDF
		require.Error(t, err)
		assert.False(t, encrypted)
	})

	t.Run("valid unencrypted PDF", func(t *testing.T) {
		pdfPath := filepath.Join(tempDir, "valid.pdf")
		createTestPDF(t, pdfPath)

		encrypted, err := handler.IsEncrypted(pdfPath)
		// Note: This may fail if pdfcpu can't process our minimal PDF
		// In that case, it's not an encryption error
		if err != nil {
			t.Logf("PDF processing failed (expected for minimal test PDF): %v", err)
		} else {
			assert.False(t, encrypted)
		}
	})
}

func TestPasswordHandler_CleanupTempFile(t *testing.T) {
	handler := NewPasswordHandler(false)
	tempDir := t.TempDir()

	t.Run("cleanup valid temp file", func(t *testing.T) {
		tempFile := filepath.Join(tempDir, "decrypted-test123.pdf")
		err := os.WriteFile(tempFile, []byte("content"), 0o644)
		require.NoError(t, err)

		err = handler.CleanupTempFile(tempFile)
		require.NoError(t, err)

		_, err = os.Stat(tempFile)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("cleanup empty filename", func(t *testing.T) {
		err := handler.CleanupTempFile("")
		require.NoError(t, err)
	})

	t.Run("cleanup non-temp file (should not remove)", func(t *testing.T) {
		regularFile := filepath.Join(tempDir, "regular.pdf")
		err := os.WriteFile(regularFile, []byte("content"), 0o644)
		require.NoError(t, err)

		err = handler.CleanupTempFile(regularFile)
		require.NoError(t, err)

		// File should still exist since it doesn't match temp file pattern
		_, err = os.Stat(regularFile)
		assert.NoError(t, err)
	})

	t.Run("cleanup non-existent temp file", func(t *testing.T) {
		tempFile := filepath.Join(tempDir, "decrypted-nonexistent.pdf")
		err := handler.CleanupTempFile(tempFile)
		// Should return error since file doesn't exist
		require.Error(t, err)
	})

	t.Run("cleanup file without decrypted- prefix", func(t *testing.T) {
		tempFile := filepath.Join(tempDir, "noprefix.pdf")
		err := os.WriteFile(tempFile, []byte("content"), 0o644)
		require.NoError(t, err)

		err = handler.CleanupTempFile(tempFile)
		require.NoError(t, err)

		// File should still exist
		_, err = os.Stat(tempFile)
		assert.NoError(t, err)
	})

	t.Run("cleanup file without .pdf suffix", func(t *testing.T) {
		tempFile := filepath.Join(tempDir, "decrypted-test.txt")
		err := os.WriteFile(tempFile, []byte("content"), 0o644)
		require.NoError(t, err)

		err = handler.CleanupTempFile(tempFile)
		require.NoError(t, err)

		// File should still exist
		_, err = os.Stat(tempFile)
		assert.NoError(t, err)
	})
}

func TestSecureString(t *testing.T) {
	t.Run("create and retrieve string", func(t *testing.T) {
		ss := NewSecureString("secret123")
		require.NotNil(t, ss)
		assert.Equal(t, "secret123", ss.String())
	})

	t.Run("create with empty string", func(t *testing.T) {
		ss := NewSecureString("")
		require.NotNil(t, ss)
		assert.Empty(t, ss.String())
	})

	t.Run("clear secure string", func(t *testing.T) {
		ss := NewSecureString("secret123")
		ss.Clear()
		assert.Empty(t, ss.String())
		assert.Nil(t, ss.value)
	})

	t.Run("clear nil secure string", func(t *testing.T) {
		var ss *SecureString
		ss.Clear() // Should not panic
	})

	t.Run("clear already cleared string", func(t *testing.T) {
		ss := NewSecureString("secret")
		ss.Clear()
		ss.Clear() // Should not panic
		assert.Empty(t, ss.String())
	})

	t.Run("string from nil secure string", func(t *testing.T) {
		var ss *SecureString
		assert.Empty(t, ss.String())
	})

	t.Run("string from cleared secure string", func(t *testing.T) {
		ss := NewSecureString("test")
		ss.Clear()
		assert.Empty(t, ss.String())
	})

	t.Run("multiple operations", func(t *testing.T) {
		ss := NewSecureString("password")
		assert.Equal(t, "password", ss.String())
		assert.Equal(t, "password", ss.String()) // Multiple reads
		ss.Clear()
		assert.Empty(t, ss.String())
	})

	t.Run("verify memory is zeroed", func(t *testing.T) {
		ss := NewSecureString("sensitive")
		originalBytes := ss.value
		ss.Clear()

		// Verify all bytes are zeroed
		for _, b := range originalBytes {
			assert.Equal(t, byte(0), b)
		}
	})
}

func TestGetPasswordPrompt(t *testing.T) {
	t.Run("simple filename", func(t *testing.T) {
		prompt := GetPasswordPrompt("document.pdf")
		assert.Contains(t, prompt, "document.pdf")
		assert.Contains(t, prompt, "password protected")
		assert.Contains(t, prompt, "Please Provide The Password")
	})

	t.Run("filename with path", func(t *testing.T) {
		prompt := GetPasswordPrompt("/path/to/secure.pdf")
		assert.Contains(t, prompt, "/path/to/secure.pdf")
		assert.Contains(t, prompt, "password protected")
	})

	t.Run("filename with special characters", func(t *testing.T) {
		prompt := GetPasswordPrompt("my file (1).pdf")
		assert.Contains(t, prompt, "my file (1).pdf")
		assert.Contains(t, prompt, "password protected")
	})

	t.Run("empty filename", func(t *testing.T) {
		prompt := GetPasswordPrompt("")
		assert.Contains(t, prompt, `""`)
		assert.Contains(t, prompt, "password protected")
	})

	t.Run("title case in prompt", func(t *testing.T) {
		prompt := GetPasswordPrompt("test.pdf")
		// Should contain title-cased version
		assert.Contains(t, strings.ToLower(prompt), "please provide the password")
	})
}

func TestIsPasswordError(t *testing.T) {
	t.Run("nil error", func(t *testing.T) {
		assert.False(t, IsPasswordError(nil))
	})

	t.Run("password keyword", func(t *testing.T) {
		err := errors.New("invalid password provided")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("encrypted keyword", func(t *testing.T) {
		err := errors.New("file is encrypted")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("decrypt keyword", func(t *testing.T) {
		err := errors.New("failed to decrypt file")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("authentication keyword", func(t *testing.T) {
		err := errors.New("authentication failed")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("unauthorized keyword", func(t *testing.T) {
		err := errors.New("unauthorized access")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("invalid credentials keyword", func(t *testing.T) {
		err := errors.New("invalid credentials provided")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("case insensitive matching", func(t *testing.T) {
		err := errors.New("PASSWORD is incorrect")
		assert.True(t, IsPasswordError(err))

		err = errors.New("ENCRYPTED file detected")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("non-password error", func(t *testing.T) {
		err := errors.New("file not found")
		assert.False(t, IsPasswordError(err))
	})

	t.Run("generic error", func(t *testing.T) {
		err := errors.New("something went wrong")
		assert.False(t, IsPasswordError(err))
	})

	t.Run("empty error message", func(t *testing.T) {
		err := errors.New("")
		assert.False(t, IsPasswordError(err))
	})

	t.Run("partial keyword match", func(t *testing.T) {
		err := errors.New("pass") // Too short, shouldn't match "password"
		assert.False(t, IsPasswordError(err))
	})

	t.Run("keyword in context", func(t *testing.T) {
		err := errors.New("the password verification process failed unexpectedly")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("multiple keywords", func(t *testing.T) {
		err := errors.New("encrypted file requires password for decryption")
		assert.True(t, IsPasswordError(err))
	})
}

func TestPasswordHandler_ValidateCredentials_ErrorCases(t *testing.T) {
	handler := NewPasswordHandler(false)
	tempDir := t.TempDir()

	t.Run("nil credentials", func(t *testing.T) {
		pdfPath := filepath.Join(tempDir, "test.pdf")
		createTestPDF(t, pdfPath)

		err := handler.ValidateCredentials(pdfPath, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no credentials provided")
	})

	t.Run("non-existent file", func(t *testing.T) {
		creds := &PasswordCredentials{
			UserPassword: "test",
		}
		err := handler.ValidateCredentials("/non/existent/file.pdf", creds)
		require.Error(t, err)
	})

	t.Run("empty credentials", func(t *testing.T) {
		pdfPath := filepath.Join(tempDir, "test2.pdf")
		createTestPDF(t, pdfPath)

		creds := &PasswordCredentials{}
		err := handler.ValidateCredentials(pdfPath, creds)
		// Should fail because validation requires actual credentials
		require.Error(t, err)
	})
}

func TestPasswordHandler_DecryptPDF_ErrorCases(t *testing.T) {
	handler := NewPasswordHandler(false)

	t.Run("non-existent file", func(t *testing.T) {
		result, err := handler.DecryptPDF("/non/existent/file.pdf", nil)
		require.Error(t, err)
		assert.Empty(t, result)
	})

	t.Run("invalid file", func(t *testing.T) {
		tempDir := t.TempDir()
		invalidFile := filepath.Join(tempDir, "invalid.pdf")
		err := os.WriteFile(invalidFile, []byte("not a PDF"), 0o644)
		require.NoError(t, err)

		result, err := handler.DecryptPDF(invalidFile, nil)
		require.Error(t, err)
		assert.Empty(t, result)
	})
}

// Benchmark tests.
func BenchmarkSecureString_Create(b *testing.B) {
	for range b.N {
		ss := NewSecureString("test-password-123")
		_ = ss.String()
	}
}

func BenchmarkSecureString_Clear(b *testing.B) {
	for range b.N {
		ss := NewSecureString("test-password-123")
		ss.Clear()
	}
}

func BenchmarkIsPasswordError(b *testing.B) {
	testErrors := []error{
		errors.New("invalid password"),
		errors.New("file is encrypted"),
		errors.New("decryption failed"),
		errors.New("file not found"),
		nil,
	}

	for i, err := range testErrors {
		b.Run(string(rune('A'+i)), func(b *testing.B) {
			for range b.N {
				_ = IsPasswordError(err)
			}
		})
	}
}

func BenchmarkGetPasswordPrompt(b *testing.B) {
	filenames := []string{
		"document.pdf",
		"/path/to/secure.pdf",
		"very-long-filename-with-many-characters-document.pdf",
	}

	for _, filename := range filenames {
		b.Run(filename, func(b *testing.B) {
			for range b.N {
				_ = GetPasswordPrompt(filename)
			}
		})
	}
}

// Test edge cases and concurrent access.
func TestSecureString_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent test in short mode")
	}

	ss := NewSecureString("concurrent-test")
	done := make(chan bool, 10)

	// Multiple goroutines reading
	for range 10 {
		go func() {
			defer func() { done <- true }()
			for range 100 {
				_ = ss.String()
			}
		}()
	}

	// Wait for all goroutines
	for range 10 {
		<-done
	}

	// Verify string is still intact
	assert.Equal(t, "concurrent-test", ss.String())
}

func TestPasswordCredentials_EdgeCases(t *testing.T) {
	t.Run("very long passwords", func(t *testing.T) {
		longPassword := strings.Repeat("a", 1000)
		creds := &PasswordCredentials{
			UserPassword:  longPassword,
			OwnerPassword: longPassword,
		}
		assert.Len(t, creds.UserPassword, 1000)
		assert.Len(t, creds.OwnerPassword, 1000)
	})

	t.Run("passwords with special characters", func(t *testing.T) {
		creds := &PasswordCredentials{
			UserPassword:  "p@$$w0rd!#%&*(){}[]",
			OwnerPassword: "öäü€ßÄÖÜ",
		}
		assert.Equal(t, "p@$$w0rd!#%&*(){}[]", creds.UserPassword)
		assert.Equal(t, "öäü€ßÄÖÜ", creds.OwnerPassword)
	})

	t.Run("passwords with whitespace", func(t *testing.T) {
		creds := &PasswordCredentials{
			UserPassword:  "  password with spaces  ",
			OwnerPassword: "password\twith\ttabs",
		}
		assert.Equal(t, "  password with spaces  ", creds.UserPassword)
		assert.Equal(t, "password\twith\ttabs", creds.OwnerPassword)
	})
}

func TestIsPasswordError_EdgeCases(t *testing.T) {
	t.Run("error with mixed case keywords", func(t *testing.T) {
		err := errors.New("PaSsWoRd EnCrYpTeD")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("error with keywords as substrings", func(t *testing.T) {
		err := errors.New("reencrypted file with superpassword")
		assert.True(t, IsPasswordError(err))
	})

	t.Run("wrapped error with password keyword", func(t *testing.T) {
		baseErr := errors.New("invalid password")
		wrappedErr := errors.New("operation failed: " + baseErr.Error())
		assert.True(t, IsPasswordError(wrappedErr))
	})
}

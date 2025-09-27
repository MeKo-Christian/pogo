package support

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/MeKo-Tech/pogo/internal/testutil"
)

// StartServer starts the OCR server with the given command.
func (testCtx *TestContext) StartServer(command string) error {
	// Parse command to extract port and other options
	if err := testCtx.parseServerCommand(command); err != nil {
		return err
	}

	// Check if port is already in use
	if testCtx.isPortInUse(testCtx.ServerPort) {
		return fmt.Errorf("port %d is already in use", testCtx.ServerPort)
	}

	// Parse command into parts
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return errors.New("empty command")
	}

	// If using bare cli name, replace with fixed bin path under project root
	if parts[0] == "pogo" {
		if root, err := testutil.GetProjectRoot(); err == nil {
			parts[0] = filepath.Join(root, "bin", "pogo")
		}
	}

	// Create command
	cmd := exec.Command(parts[0], parts[1:]...)
	cmd.Dir = testCtx.WorkingDir
	cmd.Env = append(os.Environ(), testCtx.EnvVars...)

	// Start the server process
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}

	testCtx.ServerProcess = cmd.Process

	// Wait for server to be ready
	if err := testCtx.waitForServerReady(); err != nil {
		if stopErr := testCtx.StopServer(); stopErr != nil {
			return fmt.Errorf("server failed to start and also failed to stop: %w; stop error: %w", err, stopErr)
		}
		return fmt.Errorf("server failed to start: %w", err)
	}

	return nil
}

// StopServerProcess stops the running server process.
func (testCtx *TestContext) StopServerProcess() error {
	if testCtx.ServerProcess == nil {
		return nil
	}

	// Send SIGTERM for graceful shutdown
	if err := testCtx.ServerProcess.Signal(syscall.SIGTERM); err != nil {
		// If SIGTERM fails, force kill
		if killErr := testCtx.ServerProcess.Kill(); killErr != nil {
			return fmt.Errorf("failed to kill server process: %w", killErr)
		}
	}

	// Wait for process to exit
	_, err := testCtx.ServerProcess.Wait()
	testCtx.ServerProcess = nil

	return err
}

// parseServerCommand extracts server configuration from command.
func (testCtx *TestContext) parseServerCommand(command string) error {
	parts := strings.Fields(command)

	// Default values
	testCtx.ServerPort = 8080
	testCtx.ServerHost = "localhost"

	// Parse flags
	for i, part := range parts {
		switch part {
		case "--port", "-p":
			if i+1 < len(parts) {
				port, err := strconv.Atoi(parts[i+1])
				if err != nil {
					return fmt.Errorf("invalid port: %s", parts[i+1])
				}
				testCtx.ServerPort = port
			}
		case "--host", "-H":
			if i+1 < len(parts) {
				testCtx.ServerHost = parts[i+1]
			}
		}

		// Handle --port=value format
		if strings.HasPrefix(part, "--port=") {
			portStr := strings.TrimPrefix(part, "--port=")
			port, err := strconv.Atoi(portStr)
			if err != nil {
				return fmt.Errorf("invalid port: %s", portStr)
			}
			testCtx.ServerPort = port
		}

		// Handle --host=value format
		if strings.HasPrefix(part, "--host=") {
			testCtx.ServerHost = strings.TrimPrefix(part, "--host=")
		}
	}

	return nil
}

// isPortInUse checks if a port is already in use.
func (testCtx *TestContext) isPortInUse(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf(":%d", port), time.Second)
	if err != nil {
		return false
	}
	if err := conn.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error closing connection: %v\n", err)
	}
	return true
}

// waitForServerReady waits for the server to respond to health checks.
func (testCtx *TestContext) waitForServerReady() error {
	timeout := time.Now().Add(10 * time.Second)

	for time.Now().Before(timeout) {
		if testCtx.isServerHealthy() {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}

	return errors.New("server did not become ready within timeout")
}

// isServerHealthy checks if the server responds to health endpoint.
func (testCtx *TestContext) isServerHealthy() bool {
	client := &http.Client{Timeout: time.Second}
	url := fmt.Sprintf("http://%s:%d/health", testCtx.ServerHost, testCtx.ServerPort)

	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "Error closing response body: %v\n", err)
		}
	}()

	return resp.StatusCode == http.StatusOK
}

// GetServerURL returns the base URL for the running server.
func (testCtx *TestContext) GetServerURL() string {
	// Use httptest server URL if available
	if testCtx.HTTPTestServer != nil && testCtx.HTTPTestServer.Server != nil {
		return testCtx.HTTPTestServer.Server.URL
	}
	// Fallback to process-based server
	return fmt.Sprintf("http://%s:%d", testCtx.ServerHost, testCtx.ServerPort)
}

// SendSignalToServer sends a signal to the running server.
func (testCtx *TestContext) SendSignalToServer(signal os.Signal) error {
	if testCtx.ServerProcess == nil {
		return errors.New("no server process running")
	}

	return testCtx.ServerProcess.Signal(signal)
}

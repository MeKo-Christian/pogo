package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockWebSocketConn is a mock implementation of websocket.Conn for testing.
type mockWebSocketConn struct {
	sentMessages []sentMessage
}

type sentMessage struct {
	messageType int
	data        []byte
}

func (m *mockWebSocketConn) WriteMessage(messageType int, data []byte) error {
	m.sentMessages = append(m.sentMessages, sentMessage{
		messageType: messageType,
		data:        data,
	})
	return nil
}

func (m *mockWebSocketConn) getSentMessages() []sentMessage {
	return m.sentMessages
}

func TestServer_ExtractWebSocketConfig(t *testing.T) {
	server := &Server{}

	t.Run("nil options", func(t *testing.T) {
		config := server.extractWebSocketConfig(nil)
		assert.NotNil(t, config)
		assert.Empty(t, config.Language)
		assert.Empty(t, config.DictPath)
		assert.Empty(t, config.DetModel)
		assert.Empty(t, config.RecModel)
		assert.Empty(t, config.DictLangs)
	})

	t.Run("empty options", func(t *testing.T) {
		config := server.extractWebSocketConfig(map[string]interface{}{})
		assert.NotNil(t, config)
		assert.Empty(t, config.Language)
		assert.Empty(t, config.DictPath)
		assert.Empty(t, config.DetModel)
		assert.Empty(t, config.RecModel)
		assert.Empty(t, config.DictLangs)
	})

	t.Run("string values", func(t *testing.T) {
		options := map[string]interface{}{
			"language":  "en",
			"dict":      "custom.dict",
			"det-model": "det.onnx",
			"rec-model": "rec.onnx",
		}

		config := server.extractWebSocketConfig(options)
		assert.Equal(t, "en", config.Language)
		assert.Equal(t, "custom.dict", config.DictPath)
		assert.Equal(t, "det.onnx", config.DetModel)
		assert.Equal(t, "rec.onnx", config.RecModel)
		assert.Empty(t, config.DictLangs)
	})

	t.Run("dict-langs as string", func(t *testing.T) {
		options := map[string]interface{}{
			"dict-langs": "en,de,fr",
		}

		config := server.extractWebSocketConfig(options)
		expected := []string{"en", "de", "fr"}
		assert.Equal(t, expected, config.DictLangs)
	})

	t.Run("dict-langs as string with spaces", func(t *testing.T) {
		options := map[string]interface{}{
			"dict-langs": "en, de , fr",
		}

		config := server.extractWebSocketConfig(options)
		expected := []string{"en", "de", "fr"}
		assert.Equal(t, expected, config.DictLangs)
	})

	t.Run("dict-langs as array", func(t *testing.T) {
		options := map[string]interface{}{
			"dict-langs": []interface{}{"en", "de", "fr"},
		}

		config := server.extractWebSocketConfig(options)
		expected := []string{"en", "de", "fr"}
		assert.Equal(t, expected, config.DictLangs)
	})

	t.Run("dict-langs as array with non-string values", func(t *testing.T) {
		options := map[string]interface{}{
			"dict-langs": []interface{}{"en", 123, "fr"},
		}

		config := server.extractWebSocketConfig(options)
		expected := []string{"en", "fr"} // Only strings should be included
		assert.Equal(t, expected, config.DictLangs)
	})

	t.Run("mixed valid and invalid types", func(t *testing.T) {
		options := map[string]interface{}{
			"language":   "en",
			"dict":       123, // Invalid type
			"det-model":  "det.onnx",
			"rec-model":  true, // Invalid type
			"dict-langs": "en,de",
		}

		config := server.extractWebSocketConfig(options)
		assert.Equal(t, "en", config.Language)
		assert.Empty(t, config.DictPath) // Should be empty due to invalid type
		assert.Equal(t, "det.onnx", config.DetModel)
		assert.Empty(t, config.RecModel) // Should be empty due to invalid type
		assert.Equal(t, []string{"en", "de"}, config.DictLangs)
	})
}

func TestServer_SendWebSocketResponse(t *testing.T) {
	mockConn := &mockWebSocketConn{}
	server := &Server{}

	response := WebSocketOCRResponse{
		Type:      "ocr_response",
		Status:    "completed",
		Progress:  1.0,
		RequestID: "test-request-id",
		Result:    "test result",
	}

	server.sendWebSocketResponse(mockConn, response)

	messages := mockConn.getSentMessages()
	require.Len(t, messages, 1)

	var receivedResponse WebSocketOCRResponse
	err := json.Unmarshal(messages[0].data, &receivedResponse)
	require.NoError(t, err)

	assert.Equal(t, websocket.TextMessage, messages[0].messageType)
	assert.Equal(t, response, receivedResponse)
}

func TestServer_SendWebSocketError(t *testing.T) {
	mockConn := &mockWebSocketConn{}
	server := &Server{}

	server.sendWebSocketError(mockConn, "test_error", "Test error message")

	messages := mockConn.getSentMessages()
	require.Len(t, messages, 1)

	var response WebSocketOCRResponse
	err := json.Unmarshal(messages[0].data, &response)
	require.NoError(t, err)

	assert.Equal(t, websocket.TextMessage, messages[0].messageType)
	assert.Equal(t, "error", response.Type)
	assert.Equal(t, "error", response.Status)
	assert.Equal(t, "Test error message", response.Error)
	assert.Equal(t, "test_error", response.ErrorType)
}

func TestWebSocketUpgrader(t *testing.T) {
	t.Run("check origin allows any origin", func(t *testing.T) {
		// Test that the upgrader allows connections from any origin
		allowed := upgrader.CheckOrigin(&http.Request{
			Header: http.Header{
				"Origin": []string{"http://example.com"},
			},
		})
		assert.True(t, allowed)

		allowed = upgrader.CheckOrigin(&http.Request{
			Header: http.Header{
				"Origin": []string{"https://another-domain.com"},
			},
		})
		assert.True(t, allowed)
	})

	t.Run("buffer sizes", func(t *testing.T) {
		assert.Equal(t, 1024, upgrader.ReadBufferSize)
		assert.Equal(t, 1024, upgrader.WriteBufferSize)
	})
}

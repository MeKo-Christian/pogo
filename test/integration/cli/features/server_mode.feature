Feature: OCR HTTP Server
  As a service consumer
  I want to use the OCR server API
  So that I can integrate OCR capabilities into my applications

  Background:
    Given the OCR models are available
    And the server is not already running

  Scenario: Start server with default settings
    When I start the server with "pogo serve"
    Then the server should start on port 8080
    And the health endpoint should respond with status 200
    And the models endpoint should be accessible

  Scenario: Start server on custom port
    When I start the server with "pogo serve --port 3000"
    Then the server should start on port 3000
    And the health endpoint should be accessible on port 3000

  Scenario: Start server with custom host
    When I start the server with "pogo serve --host 0.0.0.0 --port 8081"
    Then the server should start on host "0.0.0.0" and port 8081
    And the server should be accessible from external connections

  Scenario: Process image via API
    Given the server is running on port 8080
    When I POST an image to "/ocr/image"
    Then the response status should be 200
    And the response should contain OCR results
    And the response should include detected regions

  Scenario: Process image with JSON response
    Given the server is running on port 8080
    When I POST an image to "/ocr/image" with format "json"
    Then the response status should be 200
    And the response should be valid JSON-Code
    And the JSON should contain confidence scores

  Scenario: Process image with overlay response
    Given the server is running on port 8080
    When I POST an image to "/ocr/image" with overlay enabled
    Then the response status should be 200
    And the response should include overlay image data

  Scenario: Get model information
    Given the server is running on port 8080
    When I GET "/models"
    Then the response status should be 200
    And the response should list available models
    And the response should include model metadata

  Scenario: Health check endpoint
    Given the server is running on port 8080
    When I GET "/health"
    Then the response status should be 200
    And the response should indicate server is healthy

  Scenario: Upload large image
    Given the server is running on port 8080
    When I POST a large image to "/ocr/image"
    Then the response status should be 200
    And the processing should complete within timeout

  Scenario: Upload invalid image format
    Given the server is running on port 8080
    When I POST an invalid file to "/ocr/image"
    Then the response status should be 400
    And the error message should indicate invalid format

  Scenario: Upload image exceeding size limit
    Given the server is running with max upload size 1MB
    When I POST an image larger than 1MB to "/ocr/image"
    Then the response status should be 413
    And the error message should indicate file too large

  Scenario: Request timeout handling
    Given the server is running with timeout 5 seconds
    When I POST an image that takes longer than 5 seconds to process
    Then the response status should be 408
    And the error message should indicate timeout

  Scenario: CORS headers
    Given the server is running with CORS origin "*"
    When I make an OPTIONS request to "/ocr/image"
    Then the response should include CORS headers
    And Access-Control-Allow-Origin should be "*"

  Scenario: Concurrent requests
    Given the server is running on port 8080
    When I send multiple concurrent requests to "/ocr/image"
    Then all requests should be processed successfully
    And response times should be reasonable

  Scenario: Server with custom pipeline configuration
    When I start the server with "pogo serve --detect-orientation --language de"
    Then the server should start successfully
    And orientation detection should be enabled
    And German language should be configured

  Scenario: Graceful shutdown
    When I start the server with "pogo serve"
    When I send SIGTERM to the server
    Then the server should shutdown gracefully
    And pending requests should complete
    And the server should stop listening for new requests

  Scenario: Force shutdown
    When I start the server with "pogo serve"
    When I send SIGINT to the server
    Then the server should shutdown immediately
    And the process should terminate

  Scenario: Server restart after crash
    Given the server was running and crashed
    When I restart the server with "pogo serve"
    Then the server should start successfully
    And all endpoints should be functional

  Scenario: Display help for serve command
    When I run "pogo serve --help"
    Then the command should succeed
    And the output should contain usage information
    And the output should list server configuration flags

  Scenario: Server with invalid configuration
    When I run "pogo serve --port 70000"
    Then the output should contain "invalid port number"

  Scenario: Server with missing models
    Given the OCR models are not available
    When I run "pogo serve"
    Then the output should contain "dictionary not found"
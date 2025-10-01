Feature: Server API - WebSocket Endpoint
  As a client application
  I want to process OCR via WebSocket connection
  So that I can receive real-time progress updates and results

  Background:
    Given the server is running on port 8080

  Scenario: Establish WebSocket connection
    When I connect to WebSocket endpoint "/ws/ocr"
    Then the connection should be established successfully
    And the connection should remain open

  Scenario: Send image via WebSocket and receive results
    Given I have a WebSocket connection to "/ws/ocr"
    When I send an image via WebSocket
    Then I should receive a processing status message
    And I should receive OCR results via WebSocket
    And the results should be in JSON format

  Scenario: Send PDF via WebSocket and receive results
    Given I have a WebSocket connection to "/ws/ocr"
    When I send a PDF via WebSocket with page range "1-3"
    Then I should receive a processing status message
    And I should receive PDF OCR results via WebSocket
    And the results should include page information

  Scenario: WebSocket progress updates
    Given I have a WebSocket connection to "/ws/ocr"
    When I send a multi-page PDF via WebSocket
    Then I should receive progress updates for each page
    And progress updates should include percentage complete
    And the final message should contain complete results

  Scenario: WebSocket error handling
    Given I have a WebSocket connection to "/ws/ocr"
    When I send an invalid file via WebSocket
    Then I should receive an error message via WebSocket
    And the error should describe the problem
    And the connection should remain open for retry

  Scenario: WebSocket connection timeout
    Given I have a WebSocket connection to "/ws/ocr"
    When the connection is idle for longer than the timeout
    Then the connection should be closed gracefully
    And I should receive a timeout message

  Scenario: WebSocket concurrent connections
    When I establish 5 concurrent WebSocket connections
    Then all connections should be established successfully
    And each connection should be independent
    And all connections should process requests correctly

  Scenario: WebSocket message format validation
    Given I have a WebSocket connection to "/ws/ocr"
    When I send a message with correct format
    Then the message should be processed
    When I send a message with invalid format
    Then I should receive a format error message

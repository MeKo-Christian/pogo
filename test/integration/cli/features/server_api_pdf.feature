Feature: Server API - PDF Processing Endpoint
  As a client application
  I want to process PDFs via the REST API
  So that I can extract text and regions from PDF documents

  Background:
    Given the server is running on port 8080

  Scenario: Upload and process a simple PDF
    When I POST a PDF to "/ocr/pdf"
    Then the response status should be 200
    And the response should be valid JSON
    And the JSON should contain "pages" array
    And each page should have "images" array
    And the response should contain text "Sample PDF text"

  Scenario: Process PDF with page range
    When I POST a PDF to "/ocr/pdf" with pages "1-3"
    Then the response status should be 200
    And the response should be valid JSON
    And only pages 1 to 3 should be processed

  Scenario: Process PDF with specific pages
    When I POST a PDF to "/ocr/pdf" with pages "1,3,5"
    Then the response status should be 200
    And the response should be valid JSON
    And only pages 1, 3, and 5 should be processed

  Scenario: Upload PDF with JSON format
    When I POST a PDF to "/ocr/pdf" with format "json"
    Then the response status should be 200
    And the response should be valid JSON
    And the JSON should contain PDF metadata
    And the JSON should contain page results
    And the response should contain text "Sample PDF text"

  Scenario: Upload PDF with CSV format
    When I POST a PDF to "/ocr/pdf" with format "csv"
    Then the response status should be 200
    And the output should be valid CSV
    And the CSV should contain PDF-specific columns

  Scenario: Upload PDF with text format
    When I POST a PDF to "/ocr/pdf" with format "text"
    Then the response status should be 200
    And the response should contain text output
    And the output should contain page information
    And the response should contain text "Sample PDF text"

  Scenario: Upload multi-page PDF
    When I POST a multi-page PDF to "/ocr/pdf"
    Then the response status should be 200
    And the response should be valid JSON
    And the output should show 3 pages total
    And each page should have OCR results

  Scenario: Upload password-protected PDF
    When I POST a password-protected PDF to "/ocr/pdf"
    Then the response status should be 400
    And the error should mention password protection

  Scenario: Upload invalid PDF file
    When I POST an invalid file to "/ocr/pdf"
    Then the response status should be 400
    And the error should mention "not a valid PDF" or "PDF processing error"

  Scenario: Upload file that's too large
    When I POST a PDF larger than the max size to "/ocr/pdf"
    Then the response status should be 413
    And the error message should indicate file too large

  Scenario: PDF processing with custom language
    When I POST a PDF to "/ocr/pdf" with language "de"
    Then the response status should be 200
    And the response should be valid JSON
    And the results should use German language model

  Scenario: PDF processing with orientation detection
    When I POST a PDF to "/ocr/pdf" with orientation detection enabled
    Then the response status should be 200
    And the response should be valid JSON
    And orientation information should be included in results

  Scenario: Concurrent PDF processing requests
    When I send 5 concurrent PDF requests to "/ocr/pdf"
    Then all requests should be processed successfully
    And all responses should be valid JSON
    And response times should be reasonable

  Scenario: PDF processing timeout handling
    When I POST a very large PDF to "/ocr/pdf"
    And the processing takes longer than the timeout
    Then the response status should be 408
    And the error message should indicate timeout

  Scenario: Empty PDF handling
    When I POST an empty PDF to "/ocr/pdf"
    Then the response status should be 200
    And the response should be valid JSON
    And the output should indicate no pages processed

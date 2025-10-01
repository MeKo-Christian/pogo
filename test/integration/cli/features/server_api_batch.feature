Feature: Server API - Batch Processing Endpoint
  As a client application
  I want to process multiple files in a single request
  So that I can efficiently handle batch OCR tasks

  Background:
    Given the server is running on port 8080

  Scenario: Batch process multiple images
    When I POST a batch request with 3 images to "/ocr/batch"
    Then the response status should be 200
    And the response should be valid JSON
    And the JSON should contain a "results" array with 3 items
    And all batch items should have "success" field
    And the JSON should contain processing summary

  Scenario: Batch process multiple PDFs
    When I POST a batch request with 2 PDFs to "/ocr/batch"
    Then the response status should be 200
    And the response should be valid JSON
    And the JSON should contain a "results" array with 2 items
    And all PDF results should have "pages" array

  Scenario: Batch process mixed images and PDFs
    When I POST a batch request with 2 images and 1 PDF to "/ocr/batch"
    Then the response status should be 200
    And the response should be valid JSON
    And the JSON should contain a "results" array with 3 items
    And results should include both "image" and "pdf" types

  Scenario: Batch with different formats per item
    When I POST a batch request with custom formats to "/ocr/batch"
    Then the response status should be 200
    And the response should be valid JSON
    And each result should respect its requested format

  Scenario: Batch with partial failures
    When I POST a batch request with 2 valid and 1 invalid file to "/ocr/batch"
    Then the response status should be 200
    And the response should be valid JSON
    And 2 items should have success true
    And 1 item should have success false
    And failed items should include error messages

  Scenario: Batch response structure validation
    When I POST a batch request with 2 images to "/ocr/batch"
    Then the response status should be 200
    And the JSON should have "success" boolean field
    And the JSON should have "results" array field
    And the JSON should have "summary" object field
    And the summary should include total count and success count

  Scenario: Batch processing summary statistics
    When I POST a batch request with 5 files to "/ocr/batch"
    Then the response status should be 200
    And the summary should show 5 total items
    And the summary should show processing duration
    And the summary should show success and failure counts

  Scenario: Batch with oversized request
    When I POST a batch request with too many files to "/ocr/batch"
    Then the response status should be 413
    And the error should mention request too large

  Scenario: Batch processing concurrency
    When I POST a batch request with 10 files to "/ocr/batch"
    Then the response status should be 200
    And all files should be processed successfully
    And the processing should complete within reasonable time

  Scenario: Empty batch request
    When I POST an empty batch request to "/ocr/batch"
    Then the response status should be 400
    And the error should mention no files provided

Feature: Output Format Validation
  As a user
  I want OCR results in different formats
  So that I can integrate with various downstream systems

  Background:
    Given the OCR system is initialized

  Scenario: JSON output structure validation
    Given an image "testdata/images/simple_text.png"
    When I run OCR with format "json"
    Then the output should be valid JSON
    And the JSON should have an "ocr" object
    And the "ocr" object should have "regions" array
    And each region should have "box", "text", and "confidence" fields
    And each box should have "x", "y", "w", "h" coordinates
    And confidence values should be between 0 and 1

  Scenario: CSV output format validation
    Given an image "testdata/images/simple_text.png"
    When I run OCR with format "csv"
    Then the output should be valid CSV
    And the CSV should have a header row
    And the CSV should contain columns: "x", "y", "w", "h", "text", "det_conf", "rec_conf"
    And all numeric columns should contain valid numbers
    And text column should be properly quoted

  Scenario: Text format consistency
    Given an image "testdata/images/simple_text.png"
    When I run OCR with format "text"
    Then the output should be plain text
    And the output should contain only extracted text
    And the output should preserve line breaks
    And the output should have consistent encoding

  Scenario: JSON schema compliance for images
    Given multiple images with different content
    When I process them with JSON format
    Then all outputs should conform to the same JSON schema
    And optional fields should be consistently present or absent
    And array fields should never be null

  Scenario: CSV format edge cases
    Given an image with text containing commas and quotes
    When I run OCR with format "csv"
    Then special characters should be properly escaped
    And the CSV should remain parseable
    And no data should be corrupted

  Scenario: Multi-page PDF JSON structure
    Given a PDF "testdata/pdfs/multipage.pdf"
    When I run OCR with format "json"
    Then the JSON should have a "pages" array
    And each page should have a "page_number" field
    And each page should have an "images" array
    And page numbering should start at 1

  Scenario: Overlay image generation validation
    Given an image "testdata/images/simple_text.png"
    When I run OCR with overlay enabled
    Then an overlay image file should be created
    And the overlay should be in PNG format
    And the overlay should have the same dimensions as the input
    And detected regions should be visually marked

  Scenario: Format conversion consistency
    Given the same image processed in different formats
    When I extract the core OCR data from each format
    Then the detected regions should be identical
    And the text content should match exactly
    And confidence scores should be consistent

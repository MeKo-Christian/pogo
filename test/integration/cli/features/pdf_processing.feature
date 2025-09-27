Feature: PDF OCR Processing
  As a user of pogo
  I want to process PDF files
  So that I can extract text from scanned documents

  Background:
    Given the OCR models are available
    And the test PDFs are available

  Scenario: Process single PDF with default settings
    When I run "pogo pdf testdata/documents/sample.pdf"
    Then the command should succeed
    And the output should contain page information
    And the output should contain detected regions

  Scenario: Process PDF with JSON output
    When I run "pogo pdf testdata/documents/sample.pdf --format json"
    Then the command should succeed
    And the output should be valid JSON-Code
    And the JSON should contain "pages" array
    And each page should have "images" array

  Scenario: Process PDF with CSV output
    When I run "pogo pdf testdata/documents/sample.pdf --format csv"
    Then the command should succeed
    And the output should be valid CSV
    And the CSV should contain PDF-specific columns

  Scenario: Process PDF with page range
    When I run "pogo pdf testdata/documents/multipage.pdf --pages 1-3"
    Then the command should succeed
    And only pages 1 to 3 should be processed
    And the output should show 3 pages maximum

  Scenario: Process PDF with specific pages
    When I run "pogo pdf testdata/documents/multipage.pdf --pages 1,3,5"
    Then the command should succeed
    And only pages 1, 3, and 5 should be processed

  Scenario: Process PDF with confidence filtering
    When I run "pogo pdf testdata/documents/sample.pdf --confidence 0.8"
    Then the command should succeed
    And all detected regions should have confidence >= 0.8

  Scenario: Process multiple PDFs
    When I run "pogo pdf testdata/documents/sample.pdf testdata/documents/another.pdf"
    Then the command should succeed
    And the output should contain results for all PDFs

  Scenario: Save PDF results to file
    When I run "pogo pdf testdata/documents/sample.pdf --output results.json --format json"
    Then the command should succeed
    And the file "results.json" should exist
    And the file should contain valid JSON-Code

  Scenario: Process PDF with custom language
    When I run "pogo pdf testdata/documents/german.pdf --language de"
    Then the command should succeed
    And the output should contain German text extraction

  Scenario: Process password-protected PDF
    When I run "pogo pdf testdata/documents/protected.pdf"
    Then the command should fail
    And the error should mention password protection

  Scenario: Process non-existent PDF
    When I run "pogo pdf non_existent.pdf"
    Then the command should fail
    And the error should mention file not found

  Scenario: Process corrupted PDF
    When I run "pogo pdf testdata/fixtures/corrupted.pdf"
    Then the command should fail
    And the error should mention PDF processing error

  Scenario: Process empty PDF
    When I run "pogo pdf testdata/documents/empty.pdf"
    Then the command should succeed
    And the output should indicate no pages processed

  Scenario: Process PDF with no images
    When I run "pogo pdf testdata/documents/text_only.pdf"
    Then the command should succeed
    And the output should indicate no images found

  Scenario: Display help for PDF command
    When I run "pogo pdf --help"
    Then the command should succeed
    And the output should contain usage information
    And the output should list available flags

  Scenario: Process large PDF with progress indication
    When I run "pogo pdf testdata/documents/large.pdf"
    Then the command should succeed
    And processing information should be displayed
    And the command should complete within reasonable time
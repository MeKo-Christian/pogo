Feature: Image OCR Processing
  As a user of pogo
  I want to process images for text extraction
  So that I can extract text from various image formats

  Background:
    Given the OCR models are available
    And the test images are available

  Scenario: Process single image with default settings
    When I run "pogo image testdata/images/simple_text.png"
    Then the command should succeed
    And the output should contain detected text regions
    And the output should be in text format

  Scenario: Process image with JSON output
    When I run "pogo image testdata/images/simple_text.png --format json"
    Then the command should succeed
    Then the output should be valid JSON-Code
    Then the JSON should contain "regions" array
    Then the JSON should contain confidence scores

  Scenario: Process image with CSV output
    When I run "pogo image testdata/images/simple_text.png --format csv"
    Then the command should succeed
    Then the output should be valid CSV
    Then the CSV should contain coordinate columns

  Scenario: Process multiple images
    When I run "pogo image testdata/images/simple_text.png testdata/images/complex_layout.png"
    Then the command should succeed
    And the output should contain results for all images

  Scenario: Process image with confidence filtering
    When I run "pogo image testdata/images/simple_text.png --confidence 0.8"
    Then the command should succeed
    And all detected regions should have confidence >= 0.8

  Scenario: Process image with orientation detection
    When I run "pogo image testdata/images/rotated_document.png --detect-orientation"
    Then the command should succeed
    And the output should contain text from corrected orientation

  Scenario: Process image with text line orientation detection
    When I run "pogo image testdata/images/mixed_orientation.png --detect-textline"
    Then the command should succeed
    And individual text lines should be corrected for orientation

  Scenario: Save output to file
    When I run "pogo image testdata/images/simple_text.png --output results.txt"
    Then the command should succeed
    And the file "results.txt" should exist
    And the file should contain the OCR results

  Scenario: Process image with overlay generation
    When I run "pogo image testdata/images/simple_text.png --overlay-dir overlays"
    Then the command should succeed
    And the overlay image should be created in "overlays" directory
    And the overlay should show detected text regions

  Scenario: Process image with recognition confidence filtering
    When I run "pogo image testdata/images/simple_text.png --min-rec-conf 0.9"
    Then the command should succeed
    And only high-confidence recognized text should be included

  Scenario: Process image with custom language
    When I run "pogo image testdata/images/german_text.png --language de"
    Then the command should succeed
    And the output should contain German text

  Scenario: Process image with custom recognition height
    When I run "pogo image testdata/images/small_text.png --rec-height 48"
    Then the command should succeed
    And the recognizer should use 48 pixel height input

  Scenario: Process unsupported image format
    When I run "pogo image testdata/fixtures/test.txt"
    Then the command should fail
    And the error should mention "unsupported image format"

  Scenario: Process non-existent image
    When I run "pogo image non_existent.png"
    Then the command should fail
    And the error should mention "no such file"

  Scenario: Display help for image command
    When I run "pogo image --help"
    Then the command should succeed
    And the output should contain usage information
    And the output should list available flags
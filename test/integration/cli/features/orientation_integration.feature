Feature: Orientation Detection Integration
  As a user
  I want OCR to automatically detect and correct image orientation
  So that text extraction works correctly regardless of image rotation

  Background:
    Given the OCR system is initialized

  Scenario: Process image rotated 0 degrees (no rotation needed)
    Given an image "testdata/images/text_0deg.png" with text oriented at 0 degrees
    When I run OCR with orientation detection enabled
    Then the detected orientation should be 0 degrees
    And the text extraction should be accurate
    And no rotation correction should be applied

  Scenario: Process image rotated 90 degrees
    Given an image "testdata/images/text_90deg.png" with text oriented at 90 degrees
    When I run OCR with orientation detection enabled
    Then the detected orientation should be 90 degrees
    And the image should be rotated to 0 degrees
    And the text extraction should be accurate

  Scenario: Process image rotated 180 degrees
    Given an image "testdata/images/text_180deg.png" with text oriented at 180 degrees
    When I run OCR with orientation detection enabled
    Then the detected orientation should be 180 degrees
    And the image should be rotated to 0 degrees
    And the text extraction should be accurate

  Scenario: Process image rotated 270 degrees
    Given an image "testdata/images/text_270deg.png" with text oriented at 270 degrees
    When I run OCR with orientation detection enabled
    Then the detected orientation should be 270 degrees
    And the image should be rotated to 0 degrees
    And the text extraction should be accurate

  Scenario: Orientation with confidence threshold
    Given an image with ambiguous orientation
    When I run OCR with orientation threshold 0.8
    Then orientation should only be corrected if confidence >= 0.8
    And low confidence results should use heuristic fallback

  Scenario: Text-line level orientation detection
    Given an image with mixed text orientations
    When I run OCR with text-line orientation enabled
    Then each text region should be individually oriented
    And regions with different orientations should be corrected separately

  Scenario: Orientation with overlay visualization
    Given an image "testdata/images/rotated_text.png"
    When I run OCR with orientation detection and overlay enabled
    Then the overlay image should be generated
    And the overlay should show the corrected orientation
    And detected regions should be properly aligned

  Scenario: Orientation with different confidence thresholds
    Given an image "testdata/images/slightly_rotated.png"
    When I process with orientation threshold 0.5
    Then orientation correction should be applied
    When I process with orientation threshold 0.95
    Then orientation correction may be skipped due to low confidence

  Scenario: Orientation combined with language detection
    Given an image with German text at 90 degrees
    When I run OCR with orientation detection and German language
    Then the orientation should be corrected to 0 degrees
    And German characters should be recognized correctly

  Scenario: Batch processing with orientation
    Given 5 images with different orientations
    When I process them in batch with orientation detection
    Then each image should be individually oriented
    And all text extractions should be accurate

  Scenario: PDF pages with per-page orientation
    Given a PDF with pages at different orientations
    When I process the PDF with orientation detection
    Then each page should be independently oriented
    And all pages should extract text correctly

  Scenario: Orientation heuristic fallback
    Given an image where orientation model is unavailable
    When I run OCR with orientation detection
    Then the heuristic orientation detection should be used
    And basic orientation correction should still work

  Scenario: Orientation performance timing
    Given an image "testdata/images/large_rotated.png"
    When I process with orientation detection enabled
    Then orientation detection should complete within 2 seconds
    And the total processing time should be reasonable

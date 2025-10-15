Feature: Server API - Image Processing Endpoint
  As a client application
  I want to process images via the REST API
  So that I can extract text from images

  Background:
    Given the server is running on port 8080

  Scenario: Upload and process image with German language
    When I POST an image to "/ocr/image" with language "de"
    Then the response status should be 200
    And the response should contain text "ä"
    And the response should contain text "ß"


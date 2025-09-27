Feature: CLI Error Handling
  As a user of pogo
  I want clear error messages
  So that I can understand and fix issues

  Scenario: Missing input file for image command
    When I run "pogo image non_existent.png"
    Then the command should fail
    And the error should mention "file not found" or "no such file"

  Scenario: No arguments provided to image command
    When I run "pogo image"
    Then the command should fail
    And the error should mention "no input files provided"

  Scenario: Unsupported image format
    When I run "pogo image testdata/fixtures/test.txt"
    Then the command should fail
    And the error should mention "unsupported image format"

  Scenario: Invalid confidence threshold (too high)
    When I run "pogo image testdata/images/simple_text.png --confidence 1.5"
    Then the command should fail
    And the error should mention "invalid confidence" or "out of range"

  Scenario: Invalid confidence threshold (negative)
    When I run "pogo image testdata/images/simple_text.png --confidence -0.1"
    Then the command should fail
    And the error should mention "invalid confidence" or "out of range"

  Scenario: Invalid output format
    When I run "pogo image testdata/images/simple_text.png --format invalid"
    Then the command should fail
    And the error should mention "invalid format" or "unsupported format"

  Scenario: Invalid recognition height
    When I run "pogo image testdata/images/simple_text.png --rec-height -10"
    Then the command should fail
    And the error should mention "invalid height" or "negative height"

  Scenario: Missing models directory
    When I run "pogo image testdata/images/simple_text.png --models-dir /non/existent"
    Then the command should fail
    And the error should mention "model not found" or "no such file"

  Scenario: Corrupted image file
    When I run "pogo image testdata/fixtures/corrupted.png"
    Then the command should fail
    And the error should mention "failed to load" or "invalid image"

  Scenario: Permission denied on output file
    When I run "pogo image testdata/images/simple_text.png --output /root/readonly.txt"
    Then the command should fail
    And the error should mention "permission denied" or "failed to write"

  Scenario: Invalid page range for PDF
    When I run "pogo pdf testdata/documents/sample.pdf --pages 10-5"
    Then the command should fail
    And the error should mention "invalid page range"

  Scenario: Page range exceeding PDF size
    When I run "pogo pdf testdata/documents/sample.pdf --pages 100-200"
    Then the command should succeed
    But the output should indicate no pages in range

  Scenario: Missing PDF file
    When I run "pogo pdf non_existent.pdf"
    Then the command should fail
    And the error should mention "file not found" or "no such file"

  Scenario: Invalid PDF file
    When I run "pogo pdf testdata/fixtures/test.txt"
    Then the command should fail
    And the error should mention "not a valid PDF" or "PDF processing error"

  Scenario: Server port already in use
    Given a service is already running on port 8080
    When I run "pogo serve --port 8080"
    Then the command should fail
    And the error should mention "port already in use" or "address in use"

  Scenario: Invalid server port (too high)
    When I run "pogo serve --port 70000"
    Then the command should fail
    And the error should mention "invalid port" or "port out of range"

  Scenario: Invalid server port (negative)
    When I run "pogo serve --port -1"
    Then the command should fail
    And the error should mention "invalid port" or "negative port"

  Scenario: Invalid CORS origin format
    When I run "pogo serve --cors-origin 'invalid_url'"
    Then the command should succeed
    But a warning should be logged about invalid CORS format

  Scenario: Server with insufficient memory
    Given the system has very low memory
    When I run "pogo serve"
    Then the command might fail
    And the error should mention "memory" or "out of memory"

  Scenario: Invalid language code
    When I run "pogo image testdata/images/simple_text.png --language invalid_lang"
    Then the command should fail
    And the error should mention "unsupported language" or "invalid language"

  Scenario: Invalid threshold values
    When I run "pogo image testdata/images/simple_text.png --orientation-threshold 2.0"
    Then the command should fail
    And the error should mention "threshold out of range"

  Scenario: Missing detection model file
    When I run "pogo image testdata/images/simple_text.png --det-model /non/existent/model.onnx"
    Then the command should fail
    And the error should mention "model not found" or "no such file"

  Scenario: Missing recognition model file
    When I run "pogo image testdata/images/simple_text.png --rec-model /non/existent/model.onnx"
    Then the command should fail
    And the error should mention "model not found" or "no such file"

  Scenario: Invalid dictionary file
    When I run "pogo image testdata/images/simple_text.png --dict /non/existent/dict.txt"
    Then the command should fail
    And the error should mention "dictionary not found" or "no such file"

  Scenario: Image too large for processing
    When I run "pogo image testdata/images/extremely_large.png"
    Then the command might fail
    And if it fails, the error should mention "image too large" or "memory"

  Scenario: Network error during model download
    Given the network is unavailable
    When I run "pogo image testdata/images/simple_text.png"
    And models need to be downloaded
    Then the command should fail
    And the error should mention "network error" or "download failed"

  Scenario: Disk space insufficient
    Given the disk is full
    When I run "pogo image testdata/images/simple_text.png --output results.json"
    Then the command should fail
    And the error should mention "no space left" or "disk full"

  Scenario: Interrupted processing
    Given processing is in progress
    When I send SIGINT to the process
    Then the command should be interrupted
    And partial results should not be corrupted

  Scenario: Invalid overlay directory
    When I run "pogo image testdata/images/simple_text.png --overlay-dir /root/readonly"
    Then the command should fail
    And the error should mention "failed to create directory" or "permission denied"

  Scenario: Help for non-existent command
    When I run "pogo nonexistent --help"
    Then the command should fail
    And the error should suggest available commands

  Scenario: Invalid global flag
    When I run "pogo --invalid-flag image test.png"
    Then the command should fail
    And the error should mention "unknown flag"

  Scenario: Version flag
    When I run "pogo --version"
    Then the command should succeed
    And the output should contain version information

  Scenario: Help for main command
    When I run "pogo --help"
    Then the command should succeed
    And the output should list available subcommands
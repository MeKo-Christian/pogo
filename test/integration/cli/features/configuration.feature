Feature: CLI Configuration
  As a user of pogo
  I want to configure the OCR pipeline
  So that I can customize processing for my needs

  Background:
    Given the test images are available

  Scenario: Custom models directory
    Given the OCR models are available in a temporary directory
    When I run "pogo image testdata/images/simple_text.png --models-dir {temp_models_dir}"
    Then the command should succeed
    And the models should be loaded from the temporary directory

  Scenario: Language selection for English
    When I run "pogo image testdata/images/english_text.png --language en"
    Then the command should succeed
    And the English language model should be used

  Scenario: Language selection for German
    When I run "pogo image testdata/images/german_text.png --language de"
    Then the command should succeed
    And the German language model should be used

  Scenario: Custom detection model path
    Given a custom detection model exists at "/custom/det_model.onnx"
    When I run "pogo image testdata/images/simple_text.png --det-model /custom/det_model.onnx"
    Then the command should succeed
    And the custom detection model should be used

  Scenario: Custom recognition model path
    Given a custom recognition model exists at "/custom/rec_model.onnx"
    When I run "pogo image testdata/images/simple_text.png --rec-model /custom/rec_model.onnx"
    Then the command should succeed
    And the custom recognition model should be used

  Scenario: Custom dictionary files
    Given custom dictionary files exist
    When I run "pogo image testdata/images/simple_text.png --dict /custom/dict1.txt,/custom/dict2.txt"
    Then the command should succeed
    And the custom dictionaries should be merged and used

  Scenario: Verbose output mode
    When I run "pogo image testdata/images/simple_text.png --verbose"
    Then the command should succeed
    And the output should include debug information
    And timing information should be displayed

  Scenario: Detection confidence threshold configuration
    When I run "pogo image testdata/images/simple_text.png --confidence 0.9"
    Then the command should succeed
    And only regions with confidence >= 0.9 should be detected

  Scenario: Recognition confidence threshold configuration
    When I run "pogo image testdata/images/simple_text.png --min-rec-conf 0.8"
    Then the command should succeed
    And only text with recognition confidence >= 0.8 should be included

  Scenario: Recognition input height configuration
    When I run "pogo image testdata/images/small_text.png --rec-height 48"
    Then the command should succeed
    And the recognizer should use 48 pixel height input

  Scenario: Orientation detection configuration
    When I run "pogo image testdata/images/rotated_text.png --detect-orientation --orientation-threshold 0.8"
    Then the command should succeed
    And orientation detection should be enabled with threshold 0.8

  Scenario: Text line orientation detection configuration
    When I run "pogo image testdata/images/mixed_orientation.png --detect-textline --textline-threshold 0.7"
    Then the command should succeed
    And text line orientation detection should be enabled with threshold 0.7

  Scenario: Output format configuration
    When I run "pogo image testdata/images/simple_text.png --format json"
    Then the command should succeed
    And the output should be in JSON format

  Scenario: CSV output format configuration
    When I run "pogo image testdata/images/simple_text.png --format csv"
    Then the command should succeed
    And the output should be in CSV format
    And the CSV should contain proper headers

  Scenario: Output file configuration
    When I run "pogo image testdata/images/simple_text.png --output results.txt"
    Then the command should succeed
    And the results should be written to "results.txt"
    And the file should contain the OCR output

  Scenario: Overlay directory configuration
    When I run "pogo image testdata/images/simple_text.png --overlay-dir overlays"
    Then the command should succeed
    And overlay images should be created in "overlays" directory
    And the overlay images should show detected regions

  Scenario: Multiple configuration flags combined
    When I run "pogo image testdata/images/simple_text.png --language de --confidence 0.8 --format json --detect-orientation"
    Then the command should succeed
    And German language should be used
    And confidence threshold should be 0.8
    And output should be in JSON format
    And orientation detection should be enabled

  Scenario: Server configuration with custom port
    When I start the server with "pogo serve --port 3000"
    Then the server should start on port 3000
    And the server should be accessible on port 3000

  Scenario: Server configuration with custom host
    When I start the server with "pogo serve --host 0.0.0.0"
    Then the server should bind to all interfaces
    And external connections should be accepted

  Scenario: Server configuration with CORS
    When I start the server with "pogo serve --cors-origin https://example.com"
    Then the server should start successfully
    And CORS should be configured for "https://example.com"

  Scenario: Server configuration with upload size limit
    When I start the server with "pogo serve --max-upload-size 10"
    Then the server should start successfully
    And the maximum upload size should be 10MB

  Scenario: Server configuration with timeout
    When I start the server with "pogo serve --timeout 60"
    Then the server should start successfully
    And request timeout should be 60 seconds

  Scenario: Server configuration with pipeline options
    When I start the server with "pogo serve --language de --detect-orientation --min-det-conf 0.7"
    Then the server should start successfully
    And the pipeline should use German language
    And orientation detection should be enabled
    And detection confidence threshold should be 0.7

  Scenario: PDF page range configuration
    When I run "pogo pdf testdata/documents/multipage.pdf --pages 2-4"
    Then the command should succeed
    And only pages 2, 3, and 4 should be processed

  Scenario: PDF specific pages configuration
    When I run "pogo pdf testdata/documents/multipage.pdf --pages 1,3,5"
    Then the command should succeed
    And only pages 1, 3, and 5 should be processed

  Scenario: Environment variable for models directory
    Given the environment variable GO_OAR_OCR_MODELS_DIR is set to "/env/models"
    When I run "pogo image testdata/images/simple_text.png"
    Then the command should succeed
    And models should be loaded from "/env/models"

  Scenario: Command line flag overrides environment variable
    Given the environment variable GO_OAR_OCR_MODELS_DIR is set to "/env/models"
    When I run "pogo image testdata/images/simple_text.png --models-dir /cli/models"
    Then the command should succeed
    And models should be loaded from "/cli/models"

  Scenario: Invalid configuration combination
    When I run "pogo image testdata/images/simple_text.png --confidence 1.5 --min-rec-conf -0.1"
    Then the command should fail
    And the error should mention invalid configuration values

  Scenario: Help shows all configuration options
    When I run "pogo image --help"
    Then the command should succeed
    And the help should list all available flags
    And flag descriptions should be clear and helpful

  Scenario: Global help shows subcommands
    When I run "pogo --help"
    Then the command should succeed
    And the help should list all available subcommands
    And global flags should be documented

  Scenario: Version information
    When I run "pogo --version"
    Then the command should succeed
    And the output should contain version information
    And build information should be included
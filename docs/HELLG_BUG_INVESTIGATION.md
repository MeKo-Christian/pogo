# "Hellg" Bug Investigation Report

## Overview

**Status**: 🔴 CRITICAL - Under Investigation
**Discovered**: 2025-10-05
**Impact**: OCR consistently outputs "Hellg" instead of "Hello", indicating systematic character misalignment

## Symptoms

1. **Consistent Character Offset**: The character 'o' is recognized as 'g' - exactly 8 positions earlier in the dictionary
2. **Affects All Images**: Both synthetic test images and properly generated images show the same issue
3. **Dictionary Verified Correct**: `ppocrv5_dict.txt` is byte-for-byte identical to PaddleOCR's official version
4. **Whitespace Preserved**: Ideographic space (U+3000) is correctly at position 0

## Evidence

### Model Output Analysis
```
Model predicts index 16215 for the 5th character
Dictionary lookup: 16215 - 1 = 16214 (subtract 1 for CTC blank token)
Character at index 16214: 'g'
Character at index 16222: 'o' (8 positions later)
```

### Character Recognition Results
```bash
Input Image: testdata/images/simple/simple_1_Hello.png
Expected:    "Hello"
Actual:      "Hellg"

# Debug output from model:
DEBUG char 0: model_idx=16186, dict_idx=16185, char="H"
DEBUG char 1: model_idx=16213, dict_idx=16212, char="e"
DEBUG char 2: model_idx=16220, dict_idx=16219, char="l"
DEBUG char 3: model_idx=16220, dict_idx=16219, char="l"
DEBUG char 4: model_idx=16215, dict_idx=16214, char="g"  # Should be 'o'
```

### Dictionary Verification
```bash
# Verify ideographic space at position 0
$ head -n 1 models/dictionaries/ppocrv5_dict.txt | od -c
0000000 343 200 200  \n  # U+3000 ideographic space

# Verify 'g' and 'o' positions
$ sed -n '16215p' models/dictionaries/ppocrv5_dict.txt
g
$ sed -n '16223p' models/dictionaries/ppocrv5_dict.txt
o

# Verify dictionary is official version
$ diff -q /tmp/ppocrv5_dict_official.txt models/dictionaries/ppocrv5_dict.txt
# No output = files are identical
```

## Why Tests Didn't Catch This

### Critical Test Infrastructure Issues

1. **Tests Use Wrong Dictionary**
   - Integration tests explicitly use `DictionaryPPOCRKeysV1` (142 characters)
   - Should use `DictionaryPPOCRv5` (18,383 characters)
   - Located in: `internal/recognizer/inference_test.go:741`

2. **Tests Don't Validate Text Content**
   - Tests only check:
     - Command succeeds ✅
     - Output format is valid JSON/CSV ✅
     - Confidence values exist ✅
   - Tests **DO NOT** check:
     - Actual text content ❌
     - Text accuracy ❌
     - Character correctness ❌

3. **Example: Broken Test**
```go
// internal/recognizer/inference_test.go:738
func TestRecognizeBatch_Integration(t *testing.T) {
    // ...
    cfg.DictPath = models.GetDictionaryPath("", models.DictionaryPPOCRKeysV1) // ❌ Wrong dict

    // Test passes if:
    require.Len(t, results, len(regions))  // ✅ Some results returned
    assert.GreaterOrEqual(t, result.Confidence, 0.0)  // ✅ Confidence exists

    // But NEVER checks:
    // assert.Equal(t, "BATCH INTEGRATION TEST", result.Text)  // ❌ Missing!
}
```

4. **BDD Tests Are Broken**
```gherkin
# test/integration/cli/features/image_processing.feature
Scenario: Process single image with default settings
  When I run "pogo image testdata/images/simple_text.png"
  Then the command should succeed  # ✅ Passes even with wrong output
  And the output should contain detected text regions  # ✅ Just checks structure
  # Missing: And the output should contain text "Hello"  # ❌ Not validated!
```

## Investigation Plan

See [PLAN.md Phase 0.5](../PLAN.md#05-hellg-bug-investigation--critical) for detailed investigation phases.

### Quick Start Investigation

1. **Compare with PaddleOCR Python** (Highest Priority)
   ```bash
   # Install PaddleOCR
   pip install paddleocr paddlepaddle

   # Test same image
   python scripts/test_paddleocr_comparison.py testdata/images/simple/simple_1_Hello.png
   ```

2. **Verify Preprocessing**
   ```bash
   # Enable preprocessing debug output
   export POGO_DEBUG_PREPROCESSING=1
   ./bin/pogo image testdata/images/simple/simple_1_Hello.png --format text

   # Check saved preprocessed images
   ls -la /tmp/pogo_debug/
   ```

3. **Test Minimal Dictionary**
   ```bash
   # Create minimal Latin-only dictionary
   echo -e "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz" > /tmp/minimal.txt

   # Test with minimal dictionary
   ./bin/pogo image testdata/images/simple/simple_1_Hello.png --dict /tmp/minimal.txt --format text
   ```

## Hypotheses (Ordered by Likelihood)

### 1. Preprocessing Difference (70% likely)
**Hypothesis**: Our image preprocessing differs from PaddleOCR's, causing the model to receive different input

**Evidence for**:
- Same model, same dictionary, different results suggests input differences
- Image preprocessing is complex: resizing, normalization, padding, color space

**Evidence against**:
- We're using standard ONNX Runtime preprocessing
- Character offset is consistent (always 8 positions)

**Test**: Compare preprocessed tensors byte-by-byte with PaddleOCR

### 2. CTC Decoding Off-by-One Error (20% likely)
**Hypothesis**: We have an off-by-one error in how we map CTC indices to dictionary positions

**Evidence for**:
- Consistent 8-position offset suggests systematic indexing error
- CTC blank token handling is tricky (index 0 vs separate)

**Evidence against**:
- Simple `idx - 1` logic seems correct
- Would affect all characters, not just 'o'

**Test**: Manually trace through CTC decode with known indices

### 3. Model-Dictionary Incompatibility (5% likely)
**Hypothesis**: The PP-OCRv5 model we downloaded expects a different dictionary format/order

**Evidence for**:
- Model comes from external source
- Dictionary order matters for character mapping

**Evidence against**:
- Dictionary is official from PaddleOCR repository
- File is byte-identical to official version

**Test**: Test with different PP-OCRv5 model versions

### 4. Synthetic Image Artifacts (5% likely)
**Hypothesis**: Our synthetic test image generator creates images that confuse the model

**Evidence for**:
- All tested images are synthetic
- Font rendering might differ from training data

**Evidence against**:
- Multiple different images show same issue
- Even properly generated images fail

**Test**: Test with real-world photos of text

## Next Steps

1. ✅ **Document the issue** (This file)
2. 🔄 **Update PLAN.md** with investigation phases
3. ⏳ **Set up PaddleOCR comparison environment**
4. ⏳ **Run baseline comparison test**
5. ⏳ **Follow investigation plan based on baseline results**

## Team Notes

- This bug was hidden by inadequate test validation
- **All tests must be updated to validate text content** (see Phase 0.1a)
- No PR should be merged without text accuracy validation
- Testing anti-pattern: checking format instead of content

## References

- [PLAN.md Phase 0.5](../PLAN.md#05-hellg-bug-investigation--critical) - Full investigation plan
- [PLAN.md Phase 0.1a](../PLAN.md#01a-test-infrastructure-critical-fixes--high-priority) - Test fixes
- PaddleOCR Dictionary: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt
- PP-OCRv5 Model: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md

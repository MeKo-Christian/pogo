# Go OCR Implementation Plan

## Project Overview

Porting OAR-OCR from Rust to Go for inference-only OCR pipeline with text detection, recognition, and optional orientation correction. Supporting both CLI tool and server service deployment with PDF processing capabilities.

## Development Phases

---

## Phase 1: Foundation & Environment Setup (Week 1)

### 1.1 Project Infrastructure

- [x] Initialize Go module `github.com/MeKo-Tech/go-oar-ocr`
- [x] Set up project directory structure:
  ```
  go-oar-ocr/
  ├── cmd/
  │   └── ocr/
  ├── internal/
  │   ├── detector/
  │   ├── recognizer/
  │   ├── pipeline/
  │   └── utils/
  ├── pkg/
  │   └── ocr/
  ├── models/
  ├── testdata/
  ├── scripts/
  └── docs/
  ```
- [x] Configure Go modules and dependencies
- [x] Set up CI/CD pipeline (GitHub Actions)
- [x] Initialize git repository with proper .gitignore

### 1.2 Dependency Management

- [x] Add `github.com/yalue/onnxruntime_go` for ONNX Runtime bindings
- [x] Add `github.com/disintegration/imaging` for image processing
- [x] Add `github.com/spf13/cobra` for CLI framework
- [x] Add `github.com/pdfcpu/pdfcpu` for PDF processing
- [x] Add testing dependencies (testify)
- [ ] Configure ONNX Runtime shared library setup
- [ ] Create setup scripts for ONNX Runtime installation
- [ ] Verify cgo compilation works correctly

### 1.3 Model and Resource Acquisition

- [ ] Download PaddleOCR PP-OCRv4/v5 detection models:
  - [ ] Mobile detection model (ppocrv4_mobile_det.onnx ~5MB)
  - [ ] Server detection model for higher accuracy
- [ ] Download PaddleOCR recognition models:
  - [ ] English PP-OCRv4 mobile recognition (en_ppocrv4_mobile_rec.onnx ~7.7MB)
  - [ ] Character dictionary (en_dict.txt)
- [ ] Download optional orientation models:
  - [ ] Document orientation classifier (pplcnet_x1_0_doc_ori.onnx)
  - [ ] Text line orientation classifier
- [ ] Verify model compatibility with ONNX Runtime
- [ ] Create model download scripts
- [ ] Set up model versioning and management

### 1.4 Initial Testing Framework

- [ ] Create `/testdata` directory structure
- [ ] Collect sample test images:
  - [ ] Simple single-word images
  - [ ] Multi-line text documents
  - [ ] Rotated text samples
  - [ ] Scanned document snippets
- [ ] Generate synthetic test images for unit testing
- [ ] Set up benchmark test framework
- [ ] Create test utility functions
- [ ] Implement ONNX Runtime smoke test

---

## Phase 2: Image Processing Foundation (Week 2)

### 2.1 Core Image Loading

- [ ] Implement image file loading utilities
  - [ ] Support JPEG, PNG, BMP formats
  - [ ] Batch image loading functionality
  - [ ] Error handling for corrupted images
  - [ ] Memory-efficient loading for large images
- [ ] Create image format validation
- [ ] Implement image metadata extraction
- [ ] Add image dimension constraints validation
- [ ] Write unit tests for image loading

### 2.2 Image Preprocessing Pipeline

- [ ] Implement image resizing algorithms:
  - [ ] Aspect ratio preservation
  - [ ] Maximum dimension constraints (960px, 1024px)
  - [ ] Multiple of 32 dimension adjustment
  - [ ] High-quality resampling (Lanczos filter)
- [ ] Create image padding functionality:
  - [ ] Border padding to target dimensions
  - [ ] Background color configuration (black for OCR)
  - [ ] Centered image placement
- [ ] Implement image normalization:
  - [ ] RGB channel extraction from RGBA
  - [ ] Pixel value scaling (0-255 → 0-1)
  - [ ] Channel ordering (RGB → NCHW for ONNX)
- [ ] Add image quality assessment utilities
- [ ] Write comprehensive preprocessing tests

### 2.3 Tensor Conversion System

- [ ] Implement ONNX tensor creation:
  - [ ] Float32 tensor generation
  - [ ] NCHW layout conversion
  - [ ] Batch dimension handling
  - [ ] Memory layout optimization
- [ ] Create tensor validation utilities
- [ ] Implement tensor debugging tools
- [ ] Add tensor dimension verification
- [ ] Write tensor conversion unit tests

### 2.4 Image Utilities

- [ ] Implement coordinate transformation utilities
- [ ] Create image cropping functions
- [ ] Add image rotation helpers (90°, 180°, 270°)
- [ ] Implement bounding box utilities
- [ ] Create polygon/contour helper functions
- [ ] Add image visualization tools for debugging
- [ ] Write utility function tests

---

## Phase 3: Text Detection Engine (Week 3)

### 3.1 Detection Model Integration

- [ ] Create Detector struct with ONNX session management
- [ ] Implement model loading and initialization:
  - [ ] ONNX session creation
  - [ ] Input/output tensor name resolution
  - [ ] Model metadata extraction
  - [ ] Thread-safe session handling
- [ ] Add detection configuration management:
  - [ ] Threshold parameters (det_db_thresh: 0.3, det_db_box_thresh: 0.5)
  - [ ] Model path configuration
  - [ ] Runtime optimization settings
- [ ] Implement model validation
- [ ] Create detection model benchmarking
- [ ] Write model loading tests

### 3.2 Detection Inference Pipeline

- [ ] Implement detection inference method:
  - [ ] Image preprocessing integration
  - [ ] ONNX Runtime session execution
  - [ ] Output tensor retrieval
  - [ ] Error handling and validation
- [ ] Create batch inference support
- [ ] Add inference timing and profiling
- [ ] Implement memory management
- [ ] Write inference unit tests

### 3.3 Detection Post-Processing

- [ ] Implement DB (Differentiable Binarization) algorithm:
  - [ ] Binary thresholding of probability maps
  - [ ] Connected component analysis (4-connectivity)
  - [ ] Flood fill / DFS for region detection
  - [ ] Region confidence calculation
- [ ] Create contour extraction system:
  - [ ] Boundary pixel detection
  - [ ] Moore-Neighbor tracing algorithm
  - [ ] Contour simplification
  - [ ] Polygon approximation
- [ ] Implement bounding box calculation:
  - [ ] Axis-aligned bounding rectangles
  - [ ] Minimum area rectangles
  - [ ] Convex hull computation
  - [ ] Rotated rectangle detection
- [ ] Add coordinate mapping back to original image scale
- [ ] Create region filtering based on confidence thresholds
- [ ] Write comprehensive post-processing tests

### 3.4 Detection Result Management

- [ ] Define TextRegion data structures:
  - [ ] Polygon coordinates
  - [ ] Bounding box rectangles
  - [ ] Confidence scores
  - [ ] Region metadata
- [ ] Implement detection result serialization
- [ ] Create result visualization tools
- [ ] Add result validation utilities
- [ ] Write detection integration tests

---

## Phase 4: Text Recognition Engine (Week 4)

### 4.1 Recognition Model Setup

- [ ] Create Recognizer struct with ONNX session
- [ ] Implement character dictionary loading:
  - [ ] Dictionary file parsing
  - [ ] Character mapping creation
  - [ ] Unicode handling
  - [ ] Multiple language support preparation
- [ ] Add recognition model initialization:
  - [ ] Model loading and validation
  - [ ] Input dimension requirements
  - [ ] Output sequence length handling
- [ ] Create recognition configuration management
- [ ] Write model setup tests

### 4.2 Text Region Preprocessing

- [ ] Implement text region cropping:
  - [ ] Polygon-based cropping
  - [ ] Rotation handling for skewed text
  - [ ] Image extraction from detected regions
- [ ] Create recognition-specific resizing:
  - [ ] Fixed height scaling (32px or 48px)
  - [ ] Aspect ratio preservation
  - [ ] Width constraints and padding
  - [ ] Dynamic width handling
- [ ] Add recognition normalization pipeline
- [ ] Implement batch cropping for efficiency
- [ ] Write cropping and preprocessing tests

### 4.3 Recognition Inference

- [ ] Implement text recognition inference:
  - [ ] Single region processing
  - [ ] Batch processing capabilities
  - [ ] Output sequence retrieval
  - [ ] Error handling for edge cases
- [ ] Create CTC (Connectionist Temporal Classification) decoding:
  - [ ] Argmax sequence extraction
  - [ ] Repeating character collapse
  - [ ] Blank token removal
  - [ ] Character index to text mapping
- [ ] Implement confidence calculation:
  - [ ] Per-character probability averaging
  - [ ] Sequence-level confidence scoring
  - [ ] Alternative confidence metrics
- [ ] Add alternative decoding methods (beam search)
- [ ] Write recognition inference tests

### 4.4 Text Output Processing

- [ ] Create text result structures:
  - [ ] Recognized text strings
  - [ ] Character-level confidence
  - [ ] Recognition timing data
- [ ] Implement text post-processing:
  - [ ] Unicode normalization
  - [ ] Special character handling
  - [ ] Text cleaning utilities
- [ ] Add multi-language text support
- [ ] Create text validation tools
- [ ] Write text processing tests

---

## Phase 5: Pipeline Integration (Week 5)

### 5.1 OCR Pipeline Architecture

- [ ] Create main OCR pipeline struct:
  - [ ] Detector and Recognizer integration
  - [ ] Configuration management
  - [ ] Resource lifecycle management
- [ ] Implement pipeline builder pattern:
  - [ ] Model path configuration
  - [ ] Feature flag management (orientation, etc.)
  - [ ] Threshold parameter setting
- [ ] Add pipeline initialization and cleanup
- [ ] Create pipeline validation
- [ ] Write pipeline setup tests

### 5.2 End-to-End Processing

- [ ] Implement single image processing:
  - [ ] Image → Detection → Recognition → Results
  - [ ] Error propagation and handling
  - [ ] Performance monitoring
  - [ ] Memory usage optimization
- [ ] Create batch image processing:
  - [ ] Multiple image handling
  - [ ] Parallel processing options
  - [ ] Progress tracking
  - [ ] Resource management
- [ ] Add processing pipeline validation
- [ ] Implement result aggregation
- [ ] Write end-to-end integration tests

### 5.3 Result Management

- [ ] Define comprehensive OCR result structures:
  - [ ] Per-region results
  - [ ] Image-level metadata
  - [ ] Processing statistics
  - [ ] Error information
- [ ] Implement result serialization:
  - [ ] JSON output format
  - [ ] Plain text extraction
  - [ ] Structured data export
- [ ] Create result visualization tools
- [ ] Add result validation utilities
- [ ] Write result processing tests

### 5.4 Performance Optimization

- [ ] Implement memory pooling for tensors
- [ ] Add GPU acceleration support (CUDA)
- [ ] Create processing pipeline profiling
- [ ] Optimize critical path performance
- [ ] Add memory leak detection
- [ ] Implement resource monitoring
- [ ] Write performance benchmarks

---

## Phase 6: Advanced Features (Week 6)

### 6.1 Document Orientation Detection

- [ ] Integrate document orientation classifier:
  - [ ] Model loading and initialization
  - [ ] Whole-image orientation prediction
  - [ ] Confidence threshold handling
  - [ ] 0°/90°/180°/270° classification
- [ ] Implement automatic image rotation:
  - [ ] Pre-detection rotation
  - [ ] Coordinate system adjustment
  - [ ] Quality preservation during rotation
- [ ] Add orientation detection configuration
- [ ] Create orientation validation tests
- [ ] Write orientation integration tests

### 6.2 Text Line Orientation Correction

- [ ] Implement text line angle classifier:
  - [ ] Per-region orientation detection
  - [ ] Vertical text handling
  - [ ] Region-specific rotation
- [ ] Add text line rotation pipeline:
  - [ ] Individual crop rotation
  - [ ] Recognition input adjustment
  - [ ] Coordinate transformation
- [ ] Create text orientation configuration
- [ ] Write text line orientation tests

### 6.3 Document Rectification (Optional)

- [ ] Research UVDoc model integration
- [ ] Implement perspective correction:
  - [ ] Warped document detection
  - [ ] Geometric transformation
  - [ ] Quality assessment
- [ ] Add rectification configuration
- [ ] Create rectification tests
- [ ] Document rectification limitations

### 6.4 Multi-Language Support

- [ ] Extend character dictionary management:
  - [ ] Multiple dictionary loading
  - [ ] Language-specific models
  - [ ] Dynamic language switching
- [ ] Add language detection capabilities
- [ ] Implement Unicode text handling
- [ ] Create language-specific tests
- [ ] Write multi-language documentation

---

## Phase 7: PDF Processing (Week 7)

### 7.1 PDF Image Extraction

- [ ] Implement PDF parsing with pdfcpu:
  - [ ] PDF file loading and validation
  - [ ] Page iteration and processing
  - [ ] Image object extraction
  - [ ] Multiple images per page handling
- [ ] Create image decoding from PDF:
  - [ ] JPEG/PNG embedded image handling
  - [ ] Image quality assessment
  - [ ] Resolution and DPI handling
- [ ] Add PDF metadata extraction
- [ ] Implement PDF error handling
- [ ] Write PDF extraction tests

### 7.2 PDF Processing Pipeline

- [ ] Integrate PDF processing with OCR pipeline:
  - [ ] Page-by-page processing
  - [ ] Batch PDF handling
  - [ ] Progress tracking for large PDFs
- [ ] Create PDF result aggregation:
  - [ ] Per-page results
  - [ ] Document-level compilation
  - [ ] Page numbering and metadata
- [ ] Add PDF processing configuration
- [ ] Implement PDF processing optimization
- [ ] Write PDF integration tests

### 7.3 PDF Output Formatting

- [ ] Create PDF-specific result formats:
  - [ ] Page-structured JSON output
  - [ ] Searchable text extraction
  - [ ] Coordinate mapping to PDF space
- [ ] Add PDF text reconstruction:
  - [ ] Reading order detection
  - [ ] Paragraph and column handling
  - [ ] Text flow optimization
- [ ] Implement PDF validation tools
- [ ] Write PDF output tests

### 7.4 PDF Edge Cases

- [ ] Handle vector-based PDFs (text extraction vs OCR)
- [ ] Process password-protected PDFs
- [ ] Manage large PDF files efficiently
- [ ] Handle corrupted or malformed PDFs
- [ ] Add PDF processing limitations documentation
- [ ] Create comprehensive PDF test suite

---

## Phase 8: CLI and Service Interface (Week 8)

### 8.1 Command-Line Interface

- [ ] Implement CLI using Cobra framework:
  - [ ] Command structure and subcommands
  - [ ] Flag and argument parsing
  - [ ] Input validation and error handling
- [ ] Add CLI commands:
  - [ ] `ocr image` - single image processing
  - [ ] `ocr batch` - batch image processing
  - [ ] `ocr pdf` - PDF processing
  - [ ] `ocr serve` - HTTP server mode
- [ ] Implement CLI configuration:
  - [ ] Model path specification
  - [ ] Output format selection (JSON/plain text)
  - [ ] Feature flags (orientation, language)
  - [ ] Verbosity and logging levels
- [ ] Add CLI help and documentation
- [ ] Write CLI integration tests

### 8.2 Output Formatting

- [ ] Implement multiple output formats:
  - [ ] JSON structured output with coordinates
  - [ ] Plain text with reading order
  - [ ] CSV format for data processing
  - [ ] XML format for compatibility
- [ ] Create output customization options:
  - [ ] Coordinate precision control
  - [ ] Confidence threshold filtering
  - [ ] Region sorting options
- [ ] Add output validation
- [ ] Write output formatting tests

### 8.3 HTTP Server Service

- [ ] Implement HTTP server using standard library:
  - [ ] RESTful API endpoints
  - [ ] File upload handling
  - [ ] Multipart form processing
  - [ ] JSON request/response handling
- [ ] Add server endpoints:
  - [ ] `POST /ocr/image` - image OCR
  - [ ] `POST /ocr/pdf` - PDF OCR
  - [ ] `GET /health` - health check
  - [ ] `GET /models` - model information
- [ ] Implement server configuration:
  - [ ] Port and binding configuration
  - [ ] Request size limits
  - [ ] Timeout settings
  - [ ] CORS handling
- [ ] Add server middleware:
  - [ ] Request logging
  - [ ] Error handling
  - [ ] Rate limiting
  - [ ] Authentication (optional)
- [ ] Write server API tests

### 8.4 Configuration Management

- [ ] Implement configuration file support:
  - [ ] YAML/JSON configuration files
  - [ ] Environment variable override
  - [ ] Command-line flag priority
- [ ] Add configuration validation
- [ ] Create default configuration templates
- [ ] Implement configuration documentation
- [ ] Write configuration tests

---

## Phase 9: Testing and Quality Assurance (Week 9)

### 9.1 Unit Testing Suite

- [ ] Achieve >90% code coverage for core components:
  - [ ] Image processing utilities
  - [ ] Detection post-processing algorithms
  - [ ] Recognition decoding logic
  - [ ] Pipeline orchestration
- [ ] Create mock ONNX Runtime for testing
- [ ] Implement property-based testing for algorithms
- [ ] Add edge case testing
- [ ] Write performance regression tests

### 9.2 Integration Testing

- [ ] Create comprehensive integration test suite:
  - [ ] End-to-end OCR pipeline testing
  - [ ] Multi-image batch processing
  - [ ] PDF processing workflows
  - [ ] CLI command testing
  - [ ] HTTP API testing
- [ ] Add reference comparison testing:
  - [ ] Compare with PaddleOCR outputs
  - [ ] Validate against known ground truth
  - [ ] Accuracy benchmarking
- [ ] Implement automated test data generation
- [ ] Create test result visualization
- [ ] Write integration test documentation

### 9.3 Performance Testing

- [ ] Implement performance benchmarks:
  - [ ] Single image processing speed
  - [ ] Batch processing throughput
  - [ ] Memory usage profiling
  - [ ] GPU acceleration benchmarks
- [ ] Create performance regression detection
- [ ] Add load testing for server mode
- [ ] Implement resource usage monitoring
- [ ] Write performance optimization guidelines

### 9.4 Quality Assurance

- [ ] Set up continuous integration:
  - [ ] Automated testing on multiple platforms
  - [ ] Code quality checks (golangci-lint)
  - [ ] Security vulnerability scanning
  - [ ] Dependency update monitoring
- [ ] Create code review guidelines
- [ ] Implement automated performance monitoring
- [ ] Add security audit procedures
- [ ] Write quality assurance documentation

---

## Phase 10: Documentation and Deployment (Week 10)

### 10.1 Technical Documentation

- [ ] Create comprehensive API documentation:
  - [ ] Go package documentation (godoc)
  - [ ] Function and method documentation
  - [ ] Type definitions and examples
- [ ] Write architecture documentation:
  - [ ] System design overview
  - [ ] Component interaction diagrams
  - [ ] Data flow documentation
- [ ] Create performance tuning guide
- [ ] Write troubleshooting documentation
- [ ] Add contributing guidelines

### 10.2 User Documentation

- [ ] Create user manual:
  - [ ] Installation instructions
  - [ ] CLI usage examples
  - [ ] API usage guides
  - [ ] Configuration options
- [ ] Write tutorial documentation:
  - [ ] Quick start guide
  - [ ] Advanced usage scenarios
  - [ ] Integration examples
- [ ] Create FAQ and troubleshooting guide
- [ ] Add example projects and demos
- [ ] Write migration guide from other OCR tools

### 10.3 Deployment Preparation

- [ ] Create release automation:
  - [ ] Binary building for multiple platforms
  - [ ] Docker container creation
  - [ ] Package manager integration
- [ ] Implement deployment configurations:
  - [ ] Kubernetes manifests
  - [ ] Docker Compose files
  - [ ] Systemd service files
- [ ] Create deployment documentation
- [ ] Add monitoring and observability setup
- [ ] Write operational runbooks

### 10.4 Distribution and Release

- [ ] Set up GitHub releases:
  - [ ] Automated release workflows
  - [ ] Binary distribution
  - [ ] Changelog generation
- [ ] Create package distributions:
  - [ ] Go module publishing
  - [ ] Docker Hub publishing
  - [ ] Package manager submissions
- [ ] Implement update mechanisms
- [ ] Create community engagement plan
- [ ] Write project roadmap

---

## Critical Success Metrics

### Functional Requirements

- [ ] Achieve >95% text detection accuracy compared to PaddleOCR
- [ ] Maintain >90% text recognition accuracy
- [ ] Support processing of 10+ image formats
- [ ] Handle PDF files up to 100 pages efficiently
- [ ] Process images up to 10 megapixels without memory issues

### Performance Requirements

- [ ] Single image processing: <2 seconds (mobile models)
- [ ] Batch processing: >10 images/minute
- [ ] Memory usage: <500MB for standard operations
- [ ] Server response time: <3 seconds for typical images
- [ ] Startup time: <5 seconds (model loading)

### Quality Requirements

- [ ] Unit test coverage: >90%
- [ ] Integration test coverage: >80%
- [ ] Zero memory leaks in long-running operations
- [ ] No crashes on malformed input files
- [ ] Graceful degradation under resource constraints

### Deployment Requirements

- [ ] Single binary deployment (minimal dependencies)
- [ ] Cross-platform compatibility (Linux, macOS, Windows)
- [ ] Container deployment ready
- [ ] Cloud service deployment capable
- [ ] Horizontal scaling support

---

## Risk Mitigation Strategies

### Technical Risks

- **ONNX Runtime Integration Issues**
  - Mitigation: Early prototype and extensive testing
  - Fallback: Pure Go ML libraries (GoMLX, Gorgonia)

- **Performance Degradation vs Rust**
  - Mitigation: Profiling and optimization focus
  - Fallback: Selective cgo usage for critical paths

- **Model Compatibility Issues**
  - Mitigation: Comprehensive model testing
  - Fallback: Model conversion tools

### Development Risks

- **Timeline Delays**
  - Mitigation: Iterative development with working milestones
  - Buffer: Optional features can be deferred

- **Resource Constraints**
  - Mitigation: Phased approach with core functionality first
  - Scaling: Community contribution for advanced features

### Operational Risks

- **Memory Usage in Production**
  - Mitigation: Extensive memory profiling and testing
  - Monitoring: Runtime memory usage tracking

- **Model Distribution**
  - Mitigation: Automated model download and validation
  - Fallback: Bundled model packages

---

## Success Criteria and Acceptance Tests

### Phase Completion Criteria

Each phase must meet the following criteria before proceeding:

- [ ] All critical todos completed
- [ ] Unit tests passing with >90% coverage
- [ ] Integration tests demonstrating functionality
- [ ] Performance benchmarks within acceptable ranges
- [ ] Code review and quality checks passed
- [ ] Documentation updated

### Final Acceptance Criteria

- [ ] Complete OCR pipeline matching OAR-OCR functionality
- [ ] CLI tool with comprehensive feature set
- [ ] HTTP server service ready for production
- [ ] PDF processing capability
- [ ] Comprehensive test suite
- [ ] Production-ready documentation
- [ ] Cross-platform binary distribution
- [ ] Performance metrics meeting requirements

This plan provides a structured approach to implementing the Go OCR system with clear milestones, comprehensive testing, and risk mitigation strategies. Each phase builds upon the previous one while maintaining working functionality throughout the development process.

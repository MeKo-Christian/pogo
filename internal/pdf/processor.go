package pdf

import (
	"fmt"
	"image"
	"os"
	"strconv"
	"time"

	"github.com/MeKo-Tech/pogo/internal/detector"
)

// ProcessorConfig contains configuration for enhanced PDF processing.
type ProcessorConfig struct {
	// Enable vector text extraction
	EnableVectorText bool
	// Enable hybrid processing (vector + OCR)
	EnableHybrid bool
	// Quality threshold for vector text
	VectorTextQuality float64
	// Coverage threshold for preferring vector text
	VectorTextCoverage float64
	// Enable password handling
	AllowPasswords bool
	// Allow password prompts
	AllowPasswordPrompt bool
}

// DefaultProcessorConfig returns the default processor configuration.
func DefaultProcessorConfig() *ProcessorConfig {
	return &ProcessorConfig{
		EnableVectorText:    true,
		EnableHybrid:        true,
		VectorTextQuality:   0.7,
		VectorTextCoverage:  0.8,
		AllowPasswords:      true,
		AllowPasswordPrompt: false,
	}
}

// Processor handles enhanced PDF OCR processing with vector text and password support.
type Processor struct {
	detector        *detector.Detector
	config          *ProcessorConfig
	analyzer        *PageAnalyzer
	hybridProcessor *HybridProcessor
	passwordHandler *PasswordHandler
	tempFiles       []string // Track temporary files for cleanup
}

// NewProcessor creates a new PDF processor with the given detector.
func NewProcessor(det *detector.Detector) *Processor {
	return NewProcessorWithConfig(det, DefaultProcessorConfig())
}

// NewProcessorWithConfig creates a new PDF processor with custom configuration.
func NewProcessorWithConfig(det *detector.Detector, config *ProcessorConfig) *Processor {
	if config == nil {
		config = DefaultProcessorConfig()
	}

	// Create analyzer with config-based settings
	analyzerConfig := &AnalyzerConfig{
		VectorTextQualityThreshold:  config.VectorTextQuality,
		VectorTextCoverageThreshold: config.VectorTextCoverage,
		HybridModeEnabled:           config.EnableHybrid,
		OCRFallbackEnabled:          true,
		MinTextDensityForOCR:        0.1,
	}

	analyzer := NewPageAnalyzer(analyzerConfig)
	hybridProcessor := NewHybridProcessor(DefaultHybridConfig())
	passwordHandler := NewPasswordHandler(config.AllowPasswordPrompt)

	return &Processor{
		detector:        det,
		config:          config,
		analyzer:        analyzer,
		hybridProcessor: hybridProcessor,
		passwordHandler: passwordHandler,
		tempFiles:       make([]string, 0),
	}
}

// ProcessFile processes a single PDF file and returns enhanced OCR results.
func (p *Processor) ProcessFile(filename string, pageRange string) (*DocumentResult, error) {
	return p.ProcessFileWithCredentials(filename, pageRange, nil)
}

// ProcessFileWithCredentials processes a PDF file with optional password credentials.
func (p *Processor) ProcessFileWithCredentials(filename string, pageRange string, creds *PasswordCredentials) (*DocumentResult, error) {
	startTime := time.Now()

	// Handle password-protected PDFs
	workingFilename := filename
	var err error
	if p.config.AllowPasswords {
		encrypted, checkErr := p.passwordHandler.IsEncrypted(filename)
		if checkErr != nil {
			return nil, fmt.Errorf("failed to check PDF encryption: %w", checkErr)
		}

		if encrypted {
			// Try to decrypt the PDF
			workingFilename, err = p.passwordHandler.DecryptPDF(filename, creds)
			if err != nil {
				return nil, fmt.Errorf("failed to decrypt PDF: %w", err)
			}

			// Track temp file for cleanup
			if workingFilename != filename {
				p.tempFiles = append(p.tempFiles, workingFilename)
			}
		}
	}

	// Analyze pages to determine processing strategy
	var pageAnalyses map[int]*PageAnalysis
	if p.config.EnableVectorText {
		pageAnalyses, err = p.analyzer.AnalyzePages(workingFilename, pageRange)
		if err != nil {
			// Continue with OCR-only if analysis fails
			pageAnalyses = make(map[int]*PageAnalysis)
		}
	}

	// Extract images from PDF
	extractStart := time.Now()
	pageImages, err := ExtractImages(workingFilename, pageRange)
	if err != nil {
		return nil, fmt.Errorf("failed to extract images from PDF: %w", err)
	}
	extractTime := time.Since(extractStart)

	// Process each page with enhanced capabilities
	pages := make([]PageResult, 0, len(pageImages))
	var totalDetectionTime, totalVectorTime time.Duration

	// Combine pages from images and vector text analysis
	allPageNums := make(map[int]bool)
	for pageNum := range pageImages {
		allPageNums[pageNum] = true
	}
	for pageNum := range pageAnalyses {
		allPageNums[pageNum] = true
	}

	for pageNum := range allPageNums {
		pageResult, detectionTime, vectorTime, err := p.processPageEnhanced(
			pageNum, pageImages[pageNum], pageAnalyses[pageNum], workingFilename)
		if err != nil {
			return nil, fmt.Errorf("failed to process page %d: %w", pageNum, err)
		}

		if pageResult != nil {
			pages = append(pages, *pageResult)
			totalDetectionTime += detectionTime
			totalVectorTime += vectorTime
		}
	}

	totalTime := time.Since(startTime)

	result := &DocumentResult{
		Filename:   filename,
		TotalPages: len(pages),
		Pages:      pages,
		Processing: ProcessingInfo{
			ExtractionTimeMs: extractTime.Milliseconds(),
			DetectionTimeMs:  totalDetectionTime.Milliseconds(),
			TotalTimeMs:      totalTime.Milliseconds(),
			VectorTimeMs:     totalVectorTime.Milliseconds(),
		},
	}

	// Clean up temporary files
	p.cleanupTempFiles()

	return result, nil
}

// processPageEnhanced processes a single PDF page with enhanced capabilities (vector text + OCR).
func (p *Processor) processPageEnhanced(pageNum int, images []image.Image, analysis *PageAnalysis, filename string) (*PageResult, time.Duration, time.Duration, error) {
	var totalDetectionTime, totalVectorTime time.Duration

	// Determine processing strategy
	strategy := StrategyOCR // Default to OCR
	if analysis != nil {
		strategy = analysis.RecommendedStrategy
	} else if p.config.EnableVectorText {
		// Analyze this page if no analysis was provided
		pageAnalysis, err := p.analyzer.AnalyzePage(filename, pageNum)
		if err == nil {
			analysis = pageAnalysis
			strategy = analysis.RecommendedStrategy
		}
	}

	// Skip page if recommended
	if strategy == StrategySkip {
		return nil, 0, 0, nil
	}

	var pageWidth, pageHeight int
	var imageResults []ImageResult
	var vectorExtraction *TextExtraction

	// Extract vector text if strategy includes it
	if strategy == StrategyVectorText || strategy == StrategyHybrid {
		vectorStart := time.Now()
		if analysis != nil && analysis.VectorTextExtraction != nil {
			vectorExtraction = analysis.VectorTextExtraction
		} else {
			// Extract vector text for this page
			extractor := NewVectorTextExtractor(p.config.VectorTextQuality)
			pageRange := strconv.Itoa(pageNum)
			extractions, err := extractor.ExtractText(filename, pageRange)
			if err == nil {
				if ext, exists := extractions[pageNum]; exists {
					vectorExtraction = ext
				}
			}
		}
		totalVectorTime = time.Since(vectorStart)
	}

	// Process images with OCR if strategy includes it
	if strategy == StrategyOCR || strategy == StrategyHybrid || len(images) > 0 {
		imageResults = make([]ImageResult, 0, len(images))

		for i, img := range images {
			bounds := img.Bounds()
			imgWidth := bounds.Dx()
			imgHeight := bounds.Dy()

			// Update page dimensions (use largest image dimensions)
			if imgWidth > pageWidth {
				pageWidth = imgWidth
			}
			if imgHeight > pageHeight {
				pageHeight = imgHeight
			}

			// Run OCR detection on the image if strategy requires it
			if strategy == StrategyOCR || strategy == StrategyHybrid {
				detectionStart := time.Now()
				regions, err := p.detector.DetectRegions(img)
				if err != nil {
					return nil, 0, 0, fmt.Errorf("OCR detection failed for image %d: %w", i, err)
				}
				detectionTime := time.Since(detectionStart)
				totalDetectionTime += detectionTime

				// Calculate average confidence
				var totalConfidence float64
				for _, region := range regions {
					totalConfidence += region.Confidence
				}

				avgConfidence := 0.0
				if len(regions) > 0 {
					avgConfidence = totalConfidence / float64(len(regions))
				}

				imageResult := ImageResult{
					ImageIndex: i,
					Width:      imgWidth,
					Height:     imgHeight,
					Regions:    regions,
					Confidence: avgConfidence,
				}

				imageResults = append(imageResults, imageResult)
			}
		}
	}

	// Set page dimensions from vector text if no images
	if pageWidth == 0 && pageHeight == 0 && vectorExtraction != nil {
		pageWidth = int(vectorExtraction.Metadata.PageWidth)
		pageHeight = int(vectorExtraction.Metadata.PageHeight)
	}

	// Create page result
	pageResult := &PageResult{
		PageNumber: pageNum,
		Width:      pageWidth,
		Height:     pageHeight,
		Images:     imageResults,
		Processing: ProcessingInfo{
			DetectionTimeMs: totalDetectionTime.Milliseconds(),
			TotalTimeMs:     (totalDetectionTime + totalVectorTime).Milliseconds(),
			VectorTimeMs:    totalVectorTime.Milliseconds(),
		},
		Strategy:         strategy,
		VectorExtraction: vectorExtraction,
	}

	// Perform hybrid processing if applicable
	if strategy == StrategyHybrid && p.config.EnableHybrid {
		hybridResult, err := p.hybridProcessor.MergeResults(
			vectorExtraction,
			imageResults,
			float64(pageWidth),
			float64(pageHeight),
		)
		if err == nil {
			pageResult.HybridResult = hybridResult
		}
	}

	return pageResult, totalDetectionTime, totalVectorTime, nil
}

// ProcessFiles processes multiple PDF files.
func (p *Processor) ProcessFiles(filenames []string, pageRange string) ([]*DocumentResult, error) {
	return p.ProcessFilesWithCredentials(filenames, pageRange, nil)
}

// ProcessFilesWithCredentials processes multiple PDF files with optional password credentials.
func (p *Processor) ProcessFilesWithCredentials(filenames []string, pageRange string,
	creds *PasswordCredentials) ([]*DocumentResult, error) {
	results := make([]*DocumentResult, 0, len(filenames))

	for _, filename := range filenames {
		result, err := p.ProcessFileWithCredentials(filename, pageRange, creds)
		if err != nil {
			return nil, fmt.Errorf("failed to process %s: %w", filename, err)
		}
		results = append(results, result)
	}

	return results, nil
}

// cleanupTempFiles cleans up any temporary files created during processing.
func (p *Processor) cleanupTempFiles() {
	for _, tempFile := range p.tempFiles {
		if err := p.passwordHandler.CleanupTempFile(tempFile); err != nil {
			// Log error but don't fail the operation
			_ = os.Remove(tempFile) // Fallback cleanup
		}
	}
	p.tempFiles = p.tempFiles[:0] // Clear the slice
}

// SetPasswordCredentials sets default password credentials for processing.
func (p *Processor) SetPasswordCredentials(creds *PasswordCredentials) {
	p.passwordHandler.SetDefaultCredentials(creds)
}

// GetConfig returns the current processor configuration.
func (p *Processor) GetConfig() *ProcessorConfig {
	return p.config
}

// UpdateConfig updates the processor configuration.
func (p *Processor) UpdateConfig(config *ProcessorConfig) {
	if config != nil {
		p.config = config

		// Update analyzer configuration
		analyzerConfig := &AnalyzerConfig{
			VectorTextQualityThreshold:  config.VectorTextQuality,
			VectorTextCoverageThreshold: config.VectorTextCoverage,
			HybridModeEnabled:           config.EnableHybrid,
			OCRFallbackEnabled:          true,
			MinTextDensityForOCR:        0.1,
		}
		p.analyzer.UpdateConfig(analyzerConfig)
	}
}

// Close cleans up any resources used by the processor.
func (p *Processor) Close() error {
	p.cleanupTempFiles()
	return nil
}

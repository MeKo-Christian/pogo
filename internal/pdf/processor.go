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
func (p *Processor) ProcessFileWithCredentials(filename string, pageRange string,
	creds *PasswordCredentials,
) (*DocumentResult, error) {
	startTime := time.Now()

	workingFilename, err := p.handlePasswordProtection(filename, creds)
	if err != nil {
		return nil, err
	}

	pageAnalyses, err := p.analyzePagesIfEnabled(workingFilename, pageRange)
	if err != nil {
		// Continue with OCR-only if analysis fails
		pageAnalyses = make(map[int]*PageAnalysis)
	}

	pageImages, extractTime, err := p.extractImagesFromPDF(workingFilename, pageRange)
	if err != nil {
		return nil, err
	}

	pages, totalDetectionTime, totalVectorTime, err := p.processAllPages(pageImages, pageAnalyses, workingFilename)
	if err != nil {
		return nil, err
	}

	result := p.createDocumentResult(filename, pages, extractTime, totalDetectionTime, totalVectorTime, startTime)

	p.cleanupTempFiles()
	return result, nil
}

// handlePasswordProtection handles decryption of password-protected PDFs.
func (p *Processor) handlePasswordProtection(filename string, creds *PasswordCredentials) (string, error) {
	if !p.config.AllowPasswords {
		return filename, nil
	}

	encrypted, err := p.passwordHandler.IsEncrypted(filename)
	if err != nil {
		return "", fmt.Errorf("failed to check PDF encryption: %w", err)
	}

	if !encrypted {
		return filename, nil
	}

	workingFilename, err := p.passwordHandler.DecryptPDF(filename, creds)
	if err != nil {
		return "", fmt.Errorf("failed to decrypt PDF: %w", err)
	}

	if workingFilename != filename {
		p.tempFiles = append(p.tempFiles, workingFilename)
	}

	return workingFilename, nil
}

// analyzePagesIfEnabled performs page analysis if vector text is enabled.
func (p *Processor) analyzePagesIfEnabled(filename, pageRange string) (map[int]*PageAnalysis, error) {
	if !p.config.EnableVectorText {
		return make(map[int]*PageAnalysis), nil
	}

	return p.analyzer.AnalyzePages(filename, pageRange)
}

// extractImagesFromPDF extracts images from the PDF and returns timing.
func (p *Processor) extractImagesFromPDF(filename, pageRange string) (map[int][]image.Image, time.Duration, error) {
	extractStart := time.Now()
	pageImages, err := ExtractImages(filename, pageRange)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to extract images from PDF: %w", err)
	}
	extractTime := time.Since(extractStart)
	return pageImages, extractTime, nil
}

// processAllPages processes all pages combining images and analysis.
func (p *Processor) processAllPages(pageImages map[int][]image.Image, pageAnalyses map[int]*PageAnalysis,
	filename string,
) ([]PageResult, time.Duration, time.Duration, error) {
	pages := make([]PageResult, 0)
	var totalDetectionTime, totalVectorTime time.Duration

	allPageNums := p.collectAllPageNumbers(pageImages, pageAnalyses)

	for pageNum := range allPageNums {
		pageResult, detectionTime, vectorTime, err := p.processPageEnhanced(
			pageNum, pageImages[pageNum], pageAnalyses[pageNum], filename)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to process page %d: %w", pageNum, err)
		}

		if pageResult != nil {
			pages = append(pages, *pageResult)
			totalDetectionTime += detectionTime
			totalVectorTime += vectorTime
		}
	}

	return pages, totalDetectionTime, totalVectorTime, nil
}

// collectAllPageNumbers combines page numbers from images and analyses.
func (p *Processor) collectAllPageNumbers(pageImages map[int][]image.Image,
	pageAnalyses map[int]*PageAnalysis,
) map[int]bool {
	allPageNums := make(map[int]bool)
	for pageNum := range pageImages {
		allPageNums[pageNum] = true
	}
	for pageNum := range pageAnalyses {
		allPageNums[pageNum] = true
	}
	return allPageNums
}

// createDocumentResult creates the final document result with timing information.
func (p *Processor) createDocumentResult(filename string, pages []PageResult, extractTime,
	totalDetectionTime, totalVectorTime time.Duration, startTime time.Time,
) *DocumentResult {
	totalTime := time.Since(startTime)

	return &DocumentResult{
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
}

// processPageEnhanced processes a single PDF page with enhanced capabilities (vector text + OCR).
func (p *Processor) processPageEnhanced(pageNum int, images []image.Image, analysis *PageAnalysis,
	filename string,
) (*PageResult, time.Duration, time.Duration, error) {
	var totalDetectionTime, totalVectorTime time.Duration

	// Determine processing strategy
	strategy, analysis := p.determineProcessingStrategy(analysis, filename, pageNum)

	// Skip page if recommended
	if strategy == StrategySkip {
		return nil, 0, 0, nil
	}

	var pageWidth, pageHeight int
	var imageResults []ImageResult
	var vectorExtraction *TextExtraction

	// Extract vector text if strategy includes it
	vectorExtraction, totalVectorTime = p.extractVectorTextIfNeeded(strategy, analysis, pageNum, filename)

	// Process images with OCR if strategy includes it
	imageResults, pageWidth, pageHeight, totalDetectionTime = p.processImagesWithOCRIfNeeded(
		strategy, images, pageWidth, pageHeight, totalDetectionTime)

	// Set page dimensions from vector text if no images
	if pageWidth == 0 && pageHeight == 0 && vectorExtraction != nil {
		pageWidth = int(vectorExtraction.Metadata.PageWidth)
		pageHeight = int(vectorExtraction.Metadata.PageHeight)
	}

	// Create and return page result
	return p.createPageResult(pageNum, pageWidth, pageHeight, imageResults,
		totalDetectionTime, totalVectorTime, strategy, vectorExtraction)
}

// determineProcessingStrategy determines the appropriate processing strategy for the page.
func (p *Processor) determineProcessingStrategy(analysis *PageAnalysis, filename string,
	pageNum int,
) (ProcessingStrategy, *PageAnalysis) {
	strategy := StrategyOCR // Default to OCR
	currentAnalysis := analysis

	if analysis != nil {
		strategy = analysis.RecommendedStrategy
	} else if p.config.EnableVectorText {
		// Analyze this page if no analysis was provided
		pageAnalysis, err := p.analyzer.AnalyzePage(filename, pageNum)
		if err == nil {
			currentAnalysis = pageAnalysis
			strategy = currentAnalysis.RecommendedStrategy
		}
	}

	return strategy, currentAnalysis
}

// createPageResult creates the final page result with optional hybrid processing.
func (p *Processor) createPageResult(pageNum, pageWidth, pageHeight int, imageResults []ImageResult,
	totalDetectionTime, totalVectorTime time.Duration, strategy ProcessingStrategy,
	vectorExtraction *TextExtraction,
) (*PageResult, time.Duration, time.Duration, error) {
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
	creds *PasswordCredentials,
) ([]*DocumentResult, error) {
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

// extractVectorTextIfNeeded extracts vector text if the strategy requires it.
func (p *Processor) extractVectorTextIfNeeded(strategy ProcessingStrategy, analysis *PageAnalysis,
	pageNum int, filename string,
) (*TextExtraction, time.Duration) {
	if strategy != StrategyVectorText && strategy != StrategyHybrid {
		return nil, 0
	}

	vectorStart := time.Now()
	var vectorExtraction *TextExtraction

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

	return vectorExtraction, time.Since(vectorStart)
}

// processImagesWithOCRIfNeeded processes images with OCR if the strategy requires it.
func (p *Processor) processImagesWithOCRIfNeeded(strategy ProcessingStrategy, images []image.Image,
	pageWidth, pageHeight int, totalDetectionTime time.Duration,
) ([]ImageResult, int, int, time.Duration) {
	if p.shouldSkipOCRProcessing(strategy, images) {
		return nil, pageWidth, pageHeight, totalDetectionTime
	}

	imageResults := make([]ImageResult, 0, len(images))
	currentPageWidth := pageWidth
	currentPageHeight := pageHeight
	currentDetectionTime := totalDetectionTime

	for i, img := range images {
		currentPageWidth, currentPageHeight = p.updatePageDimensions(img, currentPageWidth, currentPageHeight)

		if p.shouldPerformOCRDetection(strategy) {
			imageResult, detectionTime := p.performOCRDetection(img, i)
			currentDetectionTime += detectionTime
			imageResults = append(imageResults, imageResult)
		}
	}

	return imageResults, currentPageWidth, currentPageHeight, currentDetectionTime
}

// shouldSkipOCRProcessing determines if OCR processing should be skipped.
func (p *Processor) shouldSkipOCRProcessing(strategy ProcessingStrategy, images []image.Image) bool {
	return strategy != StrategyOCR && strategy != StrategyHybrid && len(images) == 0
}

// shouldPerformOCRDetection determines if OCR detection should be performed for the given strategy.
func (p *Processor) shouldPerformOCRDetection(strategy ProcessingStrategy) bool {
	return strategy == StrategyOCR || strategy == StrategyHybrid
}

// updatePageDimensions updates page dimensions based on image bounds.
func (p *Processor) updatePageDimensions(img image.Image, currentWidth, currentHeight int) (int, int) {
	bounds := img.Bounds()
	imgWidth := bounds.Dx()
	imgHeight := bounds.Dy()

	if imgWidth > currentWidth {
		currentWidth = imgWidth
	}
	if imgHeight > currentHeight {
		currentHeight = imgHeight
	}

	return currentWidth, currentHeight
}

// performOCRDetection performs OCR detection on a single image.
func (p *Processor) performOCRDetection(img image.Image, index int) (ImageResult, time.Duration) {
	detectionStart := time.Now()
	regions, err := p.detector.DetectRegions(img)
	if err != nil {
		// Return empty result on error
		bounds := img.Bounds()
		return ImageResult{
			ImageIndex: index,
			Width:      bounds.Dx(),
			Height:     bounds.Dy(),
			Regions:    nil,
			Confidence: 0.0,
		}, time.Since(detectionStart)
	}
	detectionTime := time.Since(detectionStart)

	avgConfidence := p.calculateAverageConfidence(regions)

	bounds := img.Bounds()
	imageResult := ImageResult{
		ImageIndex: index,
		Width:      bounds.Dx(),
		Height:     bounds.Dy(),
		Regions:    regions,
		Confidence: avgConfidence,
	}

	return imageResult, detectionTime
}

// calculateAverageConfidence calculates the average confidence from detected regions.
func (p *Processor) calculateAverageConfidence(regions []detector.DetectedRegion) float64 {
	if len(regions) == 0 {
		return 0.0
	}

	var totalConfidence float64
	for _, region := range regions {
		totalConfidence += region.Confidence
	}

	return totalConfidence / float64(len(regions))
}

package pipeline

import (
	"context"
	"errors"
	"image"
)

// ProcessImages processes multiple images sequentially and returns results.
func (p *Pipeline) ProcessImages(images []image.Image) ([]*OCRImageResult, error) {
	return p.ProcessImagesContext(context.Background(), images)
}

// ProcessImagesContext processes images with context cancellation support.
func (p *Pipeline) ProcessImagesContext(ctx context.Context, images []image.Image) ([]*OCRImageResult, error) {
	if len(images) == 0 {
		return nil, errors.New("no images provided")
	}

	orientationResults, workingImages, err := p.prepareOrientation(ctx, images)
	if err != nil {
		return nil, err
	}

	return p.processImagesWithOrientation(ctx, images, orientationResults, workingImages)
}

package recognizer

import (
	"errors"
	"fmt"
	"image"
	"image/color"

	"github.com/MeKo-Tech/pogo/internal/detector"
	"github.com/MeKo-Tech/pogo/internal/mempool"
	"github.com/MeKo-Tech/pogo/internal/onnx"
	"github.com/MeKo-Tech/pogo/internal/orientation"
	"github.com/MeKo-Tech/pogo/internal/utils"
	"github.com/disintegration/imaging"
)

// CropRegionImage extracts a region from the original image using the region's polygon.
// Current implementation uses the polygon's axis-aligned bounding box. If rotateIfVertical
// is true and the cropped patch is much taller than wide, it rotates 90 degrees CCW to
// make text roughly horizontal for recognition.
func CropRegionImage(img image.Image, region detector.DetectedRegion, rotateIfVertical bool) (image.Image, bool, error) {
	if img == nil {
		return nil, false, errors.New("input image is nil")
	}
	if len(region.Polygon) == 0 {
		// Fallback to box if polygon missing
		if region.Box.Width() <= 0 || region.Box.Height() <= 0 {
			return nil, false, errors.New("region polygon and box are empty")
		}
		patch := utils.CropImageBox(img, region.Box)
		rotated := false
		if rotateIfVertical {
			b := patch.Bounds()
			if b.Dy() > int(float64(b.Dx())*1.2) {
				patch = utils.Rotate90(patch)
				rotated = true
			}
		}
		return patch, rotated, nil
	}

	// Use polygon AABB for now; future: perspective/rotated rect crop
	box := utils.BoundingBox(region.Polygon)
	patch := utils.CropImageBox(img, box)
	rotated := false
	if rotateIfVertical {
		b := patch.Bounds()
		if b.Dy() > int(float64(b.Dx())*1.2) {
			patch = utils.Rotate90(patch)
			rotated = true
		}
	}
	return patch, rotated, nil
}

// CropRegionImageWithOrienter is like CropRegionImage but uses a per-region
// orientation classifier to decide rotation. The classifier returns the angle to
// apply to make text upright. If the classifier is nil, falls back to rotateIfVertical.
func CropRegionImageWithOrienter(img image.Image, region detector.DetectedRegion, cls *orientation.Classifier, rotateIfVertical bool) (image.Image, bool, error) {
	patch, _, err := CropRegionImage(img, region, false)
	if err != nil {
		return nil, false, err
	}
	if cls == nil {
		// Fallback to aspect ratio heuristic if requested
		if rotateIfVertical {
			b := patch.Bounds()
			if b.Dy() > int(float64(b.Dx())*1.5) {
				return utils.Rotate90(patch), true, nil
			}
		}
		return patch, false, nil
	}
	res, err := cls.Predict(patch)
	if err != nil {
		return patch, false, nil //nolint:nilerr // Graceful degradation when prediction fails
	}
	switch res.Angle {
	case 90:
		return utils.Rotate90(patch), true, nil
	case 180:
		return utils.Rotate180(patch), true, nil
	case 270:
		return utils.Rotate270(patch), true, nil
	default:
		// If classifier didn't recommend rotation, optionally use a simple AR heuristic
		b := patch.Bounds()
		if b.Dy() > int(float64(b.Dx())*1.2) {
			return utils.Rotate90(patch), true, nil
		}
		return patch, false, nil
	}
}

// ResizeForRecognition scales an image to a fixed target height while preserving
// aspect ratio. If padToMultiple > 0, the width is padded with black pixels to the
// next multiple. If maxWidth > 0, the width is clamped to maxWidth.
func ResizeForRecognition(img image.Image, targetHeight, maxWidth, padToMultiple int) (image.Image, int, int, error) {
	if img == nil {
		return nil, 0, 0, errors.New("input image is nil")
	}
	if targetHeight <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid targetHeight: %d", targetHeight)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w == 0 || h == 0 {
		return imaging.New(0, 0, color.Black), 0, 0, nil
	}

	scale := float64(targetHeight) / float64(h)
	newW := int(float64(w) * scale)
	if newW < 1 {
		newW = 1
	}

	// Clamp width if needed
	if maxWidth > 0 && newW > maxWidth {
		newW = maxWidth
	}

	// Resize with Lanczos filter
	resized := imaging.Resize(img, newW, targetHeight, imaging.Lanczos)

	// Pad to multiple if requested
	outW := newW
	if padToMultiple > 0 {
		rem := newW % padToMultiple
		if rem != 0 {
			outW = newW + (padToMultiple - rem)
		}
	}
	if outW == newW {
		return resized, outW, targetHeight, nil
	}

	// Create padded canvas and place resized at left
	canvas := imaging.New(outW, targetHeight, color.Black)
	canvas = imaging.Paste(canvas, resized, image.Pt(0, 0))
	return canvas, outW, targetHeight, nil
}

// NormalizeForRecognition converts an image to a float32 NCHW tensor in [0,1].
func NormalizeForRecognition(img image.Image) (onnx.Tensor, error) {
	data, w, h, err := utils.NormalizeImage(img)
	if err != nil {
		return onnx.Tensor{}, err
	}
	return onnx.NewImageTensor(data, 3, h, w)
}

// NormalizeForRecognitionWithPool normalizes using a reusable buffer pool.
// Caller should return the provided buffer via mempool.PutFloat32 after it is no longer used.
func NormalizeForRecognitionWithPool(img image.Image) (onnx.Tensor, []float32, error) {
	// Estimate required size from bounds
	b := img.Bounds()
	need := 3 * b.Dx() * b.Dy()
	buf := mempool.GetFloat32(need)
	data, w, h, err := utils.NormalizeImageIntoBuffer(img, buf)
	if err != nil {
		mempool.PutFloat32(buf)
		return onnx.Tensor{}, nil, err
	}
	ten, err := onnx.NewImageTensor(data, 3, h, w)
	if err != nil {
		mempool.PutFloat32(buf)
		return onnx.Tensor{}, nil, err
	}
	return ten, buf, nil
}

// BatchCropRegions crops multiple detected regions from a single image.
// Returns a slice of cropped patches in the same order as regions.
func BatchCropRegions(img image.Image, regions []detector.DetectedRegion, rotateIfVertical bool) ([]image.Image, []bool, error) {
	if img == nil {
		return nil, nil, errors.New("input image is nil")
	}
	patches := make([]image.Image, 0, len(regions))
	rotated := make([]bool, 0, len(regions))
	for _, r := range regions {
		patch, rot, err := CropRegionImage(img, r, rotateIfVertical)
		if err != nil {
			return nil, nil, err
		}
		patches = append(patches, patch)
		rotated = append(rotated, rot)
	}
	return patches, rotated, nil
}

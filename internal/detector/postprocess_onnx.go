package detector

import (
	"image"
)

// DetectRegions runs detection inference and post-processes regions using the
// configured DB thresholds, returning regions scaled to the original image size.
func (d *Detector) DetectRegions(img image.Image) ([]DetectedRegion, error) {
	res, err := d.RunInference(img)
	if err != nil {
		return nil, err
	}
	var regs []DetectedRegion
	opts := PostProcessOptions{UseMinAreaRect: d.config.PolygonMode != "contour"}
	if d.config.UseNMS {
		// Choose NMS method based on configuration
		switch d.config.NMSMethod {
		case "linear", "gaussian":
			regs = PostProcessDBWithOptions(res.ProbabilityMap, res.Width, res.Height,
				d.config.DbThresh, d.config.DbBoxThresh, opts)
			// Apply Soft-NMS on the results
			regs = SoftNonMaxSuppression(regs, d.config.NMSMethod, d.config.NMSThreshold,
				d.config.SoftNMSSigma, d.config.SoftNMSThresh)
		default:
			// Hard NMS as before
			regs = PostProcessDBWithNMSOptions(res.ProbabilityMap, res.Width, res.Height,
				d.config.DbThresh, d.config.DbBoxThresh, d.config.NMSThreshold, opts)
		}
	} else {
		regs = PostProcessDBWithOptions(res.ProbabilityMap, res.Width, res.Height,
			d.config.DbThresh, d.config.DbBoxThresh, opts)
	}
	regs = ScaleRegionsToOriginal(regs, res.Width, res.Height, res.OriginalWidth, res.OriginalHeight)
	return regs, nil
}

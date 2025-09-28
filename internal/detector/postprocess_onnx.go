package detector

import (
	"image"
	"log/slog"
)

// DetectRegions runs detection inference and post-processes regions using the
// configured DB thresholds, returning regions scaled to the original image size.
func (d *Detector) DetectRegions(img image.Image) ([]DetectedRegion, error) {
	res, err := d.RunInference(img)
	if err != nil {
		return nil, err
	}

	// Apply morphological operations to the probability map if configured
	probMap := res.ProbabilityMap
	if d.config.Morphology.Operation != MorphNone {
		slog.Debug("Applying morphological operations",
			"operation", d.config.Morphology.Operation,
			"kernel_size", d.config.Morphology.KernelSize,
			"iterations", d.config.Morphology.Iterations)
		probMap = ApplyMorphologicalOperation(probMap, res.Width, res.Height, d.config.Morphology)
	}

	// Calculate adaptive thresholds if enabled
	dbThresh := d.config.DbThresh
	boxThresh := d.config.DbBoxThresh
	if d.config.AdaptiveThresholds.Enabled {
		adaptiveThresh := CalculateAdaptiveThresholds(probMap, res.Width, res.Height, d.config.AdaptiveThresholds)
		dbThresh = adaptiveThresh.DbThresh
		boxThresh = adaptiveThresh.BoxThresh
		slog.Debug("Using adaptive thresholds",
			"method", adaptiveThresh.Method,
			"original_db_thresh", d.config.DbThresh,
			"adaptive_db_thresh", dbThresh,
			"original_box_thresh", d.config.DbBoxThresh,
			"adaptive_box_thresh", boxThresh,
			"confidence", adaptiveThresh.Confidence,
			"mean_prob", adaptiveThresh.Statistics.Mean,
			"std_dev", adaptiveThresh.Statistics.StdDev,
			"dynamic_range", adaptiveThresh.Statistics.DynamicRange,
			"bimodality", adaptiveThresh.Statistics.BimodalityIndex)
	}

	var regs []DetectedRegion
	opts := PostProcessOptions{UseMinAreaRect: d.config.PolygonMode != "contour"}
	if d.config.UseNMS {
		// Choose NMS method based on configuration
		switch d.config.NMSMethod {
		case "linear", "gaussian":
			slog.Debug("Using Soft-NMS for region filtering",
				"method", d.config.NMSMethod,
				"iou_threshold", d.config.NMSThreshold,
				"sigma", d.config.SoftNMSSigma)
			regs = PostProcessDBWithOptions(probMap, res.Width, res.Height,
				dbThresh, boxThresh, opts)
			// Apply Soft-NMS on the results
			regs = SoftNonMaxSuppression(regs, d.config.NMSMethod, d.config.NMSThreshold,
				d.config.SoftNMSSigma, d.config.SoftNMSThresh)
		default:
			// Check for adaptive NMS features
			switch {
			case d.config.UseAdaptiveNMS:
				slog.Debug("Using Adaptive NMS for region filtering",
					"base_threshold", d.config.NMSThreshold,
					"scale_factor", d.config.AdaptiveNMSScale)
				regs = PostProcessDBWithOptions(probMap, res.Width, res.Height,
					dbThresh, boxThresh, opts)
				regs = AdaptiveNonMaxSuppression(regs, d.config.NMSThreshold, d.config.AdaptiveNMSScale)
			case d.config.SizeAwareNMS:
				slog.Debug("Using Size-Aware NMS for region filtering",
					"base_threshold", d.config.NMSThreshold,
					"size_scale_factor", d.config.SizeNMSScaleFactor,
					"min_size", d.config.MinRegionSize,
					"max_size", d.config.MaxRegionSize)
				regs = PostProcessDBWithOptions(probMap, res.Width, res.Height,
					dbThresh, boxThresh, opts)
				regs = SizeAwareNonMaxSuppression(regs, d.config.NMSThreshold, d.config.SizeNMSScaleFactor,
					d.config.MinRegionSize, d.config.MaxRegionSize)
			default:
				slog.Debug("Using Hard-NMS for region filtering", "iou_threshold", d.config.NMSThreshold)
				// Hard NMS as before
				regs = PostProcessDBWithNMSOptions(probMap, res.Width, res.Height,
					dbThresh, boxThresh, d.config.NMSThreshold, opts)
			}
		}
	} else {
		slog.Debug("NMS disabled, using DB post-processing only")
		regs = PostProcessDBWithOptions(probMap, res.Width, res.Height,
			dbThresh, boxThresh, opts)
	}
	regs = ScaleRegionsToOriginal(regs, res.Width, res.Height, res.OriginalWidth, res.OriginalHeight)
	slog.Debug("Post-processing completed", "raw_regions", len(regs))
	return regs, nil
}

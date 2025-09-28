package rectify

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/MeKo-Tech/pogo/internal/utils"
	onnxrt "github.com/yalue/onnxruntime_go"
)

// Rectifier runs the UVDoc model and prepares data for perspective correction.
// Minimal CPU-only version: runs the model and currently returns the original image.
type Rectifier struct {
	cfg        Config
	session    *onnxrt.DynamicAdvancedSession
	inputInfo  onnxrt.InputOutputInfo
	outputInfo onnxrt.InputOutputInfo
}

// New creates a rectifier. If disabled in config, returns a stub (no session).
func New(cfg Config) (*Rectifier, error) {
	r := &Rectifier{cfg: cfg}
	if !cfg.Enabled {
		return r, nil
	}

	if err := validateModelFile(cfg.ModelPath); err != nil {
		return nil, err
	}

	session, inputInfo, outputInfo, err := createONNXSession(cfg)
	if err != nil {
		return nil, err
	}

	r.session = session
	r.inputInfo = inputInfo
	r.outputInfo = outputInfo
	return r, nil
}

func validateModelFile(modelPath string) error {
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("rectify model not found: %s", modelPath)
	}
	return nil
}

// Close releases ONNX resources.
func (r *Rectifier) Close() {
	if r == nil || r.session == nil {
		return
	}
	_ = r.session.Destroy()
	r.session = nil
}

// Apply runs rectification. Minimal version returns the original image while
// exercising the model to validate the path and future integration hooks.
func (r *Rectifier) Apply(img image.Image) (image.Image, error) {
	if r == nil || !r.cfg.Enabled || r.session == nil {
		return img, nil
	}
	if img == nil {
		return nil, errors.New("nil image")
	}

	// Prepare input image
	resized, err := utils.ResizeImage(img, utils.DefaultImageConstraints())
	if err != nil {
		return img, err
	}

	// Run model inference
	outData, oh, ow, err := r.runModelInference(resized)
	if err != nil {
		return img, err
	}

	// Process mask and find rectangle
	rect, valid := r.processMaskAndFindRectangle(outData, oh, ow)
	if !valid {
		return img, nil
	}

	// Transform and warp
	return r.transformAndWarpImage(img, resized, rect)
}

// runModelInference runs the ONNX model and returns the output tensor data.
func (r *Rectifier) runModelInference(resized image.Image) ([]float32, int, int, error) {
	data, w, h, err := r.normalizeAndValidateImage(resized)
	if err != nil {
		return nil, 0, 0, err
	}

	input, err := r.createInputTensor(data, w, h)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = input.Destroy() }()

	output, err := r.runInference(input)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = output.Destroy() }()

	return r.extractOutputData(output)
}

// normalizeAndValidateImage normalizes the image and validates dimensions.
func (r *Rectifier) normalizeAndValidateImage(resized image.Image) ([]float32, int, int, error) {
	data, w, h, err := utils.NormalizeImage(resized)
	if err != nil || w <= 0 || h <= 0 {
		return nil, 0, 0, err
	}
	return data, w, h, nil
}

// createInputTensor creates the input tensor for the model.
func (r *Rectifier) createInputTensor(data []float32, w, h int) (onnxrt.Value, error) {
	return onnxrt.NewTensor(onnxrt.NewShape(1, 3, int64(h), int64(w)), data)
}

// runInference runs the model inference.
func (r *Rectifier) runInference(input onnxrt.Value) (onnxrt.Value, error) {
	outs := []onnxrt.Value{nil}
	if err := r.session.Run([]onnxrt.Value{input}, outs); err != nil {
		return nil, err
	}
	if len(outs) == 0 || outs[0] == nil {
		return nil, errors.New("no output from model")
	}
	return outs[0], nil
}

// extractOutputData extracts and validates the output tensor data.
func (r *Rectifier) extractOutputData(output onnxrt.Value) ([]float32, int, int, error) {
	t, ok := output.(*onnxrt.Tensor[float32])
	if !ok {
		return nil, 0, 0, errors.New("invalid output tensor type")
	}

	shape := t.GetShape()
	if len(shape) != 4 || shape[1] < 3 {
		return nil, 0, 0, errors.New("unexpected output shape")
	}

	oh, ow := int(shape[2]), int(shape[3])
	return t.GetData(), oh, ow, nil
}

// processMaskAndFindRectangle extracts mask points and finds the minimum area rectangle.
func (r *Rectifier) processMaskAndFindRectangle(outData []float32, oh, ow int) ([]utils.Point, bool) {
	// Collect mask-positive points in resized coordinates.
	thr := r.cfg.MaskThreshold
	pts := make([]utils.Point, 0, (oh*ow)/8)
	mask := outData[2*oh*ow : 3*oh*ow]
	for y := range oh {
		row := y * ow
		for x := range ow {
			if float64(mask[row+x]) >= thr {
				pts = append(pts, utils.Point{X: float64(x), Y: float64(y)})
			}
		}
	}

	coverage := float64(len(pts)) / float64(oh*ow)
	if coverage < r.cfg.MinMaskCoverage || len(pts) < 100 {
		return nil, false
	}

	// Find minimum-area rectangle in resized space
	rect := utils.MinimumAreaRectangle(pts)
	if len(rect) != 4 {
		return nil, false
	}

	if r.cfg.DebugDir != "" {
		// Dump mask visualization (resized space)
		_ = dumpMaskPNG(r.cfg.DebugDir, mask, ow, oh, thr)
	}

	// Validate rectangle
	if !r.validateRectangle(rect, oh, ow) {
		return nil, false
	}

	return rect, true
}

// validateRectangle checks if the rectangle meets quality criteria.
func (r *Rectifier) validateRectangle(rect []utils.Point, oh, ow int) bool {
	// Gating based on rect area and aspect ratio in resized space
	rw0 := hypot(rect[1], rect[0])
	rw1 := hypot(rect[2], rect[3])
	rh0 := hypot(rect[3], rect[0])
	rh1 := hypot(rect[2], rect[1])
	ravgW := (rw0 + rw1) * 0.5
	ravgH := (rh0 + rh1) * 0.5

	if ravgW <= 1 || ravgH <= 1 {
		return false
	}

	rArea := ravgW * ravgH
	imgArea := float64(ow * oh)
	if rArea/imgArea < r.cfg.MinRectAreaRatio {
		return false
	}

	ar := ravgW / ravgH
	if ar < r.cfg.MinRectAspect || ar > r.cfg.MaxRectAspect {
		return false
	}

	return true
}

// transformAndWarpImage transforms coordinates and warps the image.
func (r *Rectifier) transformAndWarpImage(img, resized image.Image, rect []utils.Point) (image.Image, error) {
	// Scale rect points back to original image coordinates
	rb := resized.Bounds()
	ib := img.Bounds()
	sx := float64(ib.Dx()) / float64(rb.Dx())
	sy := float64(ib.Dy()) / float64(rb.Dy())
	srcQuad := make([]utils.Point, 4)
	for i := range 4 {
		srcQuad[i] = utils.Point{X: rect[i].X * sx, Y: rect[i].Y * sy}
	}

	if r.cfg.DebugDir != "" {
		_ = dumpOverlayPNG(r.cfg.DebugDir, img, srcQuad)
	}

	// Determine output dimensions based on quad edges
	w0 := hypot(srcQuad[1], srcQuad[0])
	w1 := hypot(srcQuad[2], srcQuad[3])
	h0 := hypot(srcQuad[3], srcQuad[0])
	h1 := hypot(srcQuad[2], srcQuad[1])
	avgW := (w0 + w1) * 0.5
	avgH := (h0 + h1) * 0.5

	if avgW <= 1 || avgH <= 1 {
		return img, nil
	}

	targetH := r.cfg.OutputHeight
	if targetH <= 0 {
		targetH = 1024
	}
	targetW := int((avgW / avgH) * float64(targetH))

	// Round to multiples of 32 to be detector-friendly
	targetW = (targetW / 32) * 32
	targetH = (targetH / 32) * 32
	if targetW < 32 {
		targetW = 32
	}
	if targetH < 32 {
		targetH = 32
	}

	dst := warpPerspective(img, srcQuad, targetW, targetH)
	if dst == nil {
		return img, nil
	}

	if r.cfg.DebugDir != "" {
		_ = dumpComparePNG(r.cfg.DebugDir, img, srcQuad, dst)
	}

	return dst, nil
}

// hypot returns Euclidean distance between points a and b.
func hypot(a, b utils.Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

// warpPerspective warps the quadrilateral region srcQuad from src into a
// target rectangle of size dstW x dstH using inverse homography + bilinear sampling.
func warpPerspective(src image.Image, srcQuad []utils.Point, dstW, dstH int) image.Image {
	if len(srcQuad) != 4 || dstW <= 0 || dstH <= 0 {
		return nil
	}

	// Build homography from dst rect to src quad. dst corners in CCW: (0,0),(W-1,0),(W-1,H-1),(0,H-1)
	d0 := utils.Point{X: 0, Y: 0}
	d1 := utils.Point{X: float64(dstW - 1), Y: 0}
	d2 := utils.Point{X: float64(dstW - 1), Y: float64(dstH - 1)}
	d3 := utils.Point{X: 0, Y: float64(dstH - 1)}
	H, ok := computeHomography(
		[4]utils.Point{d0, d1, d2, d3},
		[4]utils.Point{srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3]},
	)
	if !ok {
		return nil
	}

	// Generate destination image
	out := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	// Precompute bounds
	sb := src.Bounds()
	for y := range dstH {
		for x := range dstW {
			// Map (x,y,1) via H to source coords
			sx, sy := applyHomography(H, float64(x), float64(y))
			// Bilinear sample
			cr := bilinearSample(src, sx+float64(sb.Min.X), sy+float64(sb.Min.Y))
			out.Set(x, y, cr)
		}
	}

	return out
}

// computeHomography computes 3x3 matrix H mapping p[i] -> q[i]. Returns H as [9]float64.
func computeHomography(p, q [4]utils.Point) ([9]float64, bool) {
	// Build 8x8 system A*h = b for the 8 unknowns (h00..h21), h22=1.
	A := [8][8]float64{}
	b := [8]float64{}
	for i := range 4 {
		X, Y := p[i].X, p[i].Y
		x, y := q[i].X, q[i].Y
		r := 2 * i
		// x' = (h00 X + h01 Y + h02)/(h20 X + h21 Y + 1)
		A[r][0] = X
		A[r][1] = Y
		A[r][2] = 1
		A[r][3] = 0
		A[r][4] = 0
		A[r][5] = 0
		A[r][6] = -X * x
		A[r][7] = -Y * x
		b[r] = x

		// y' = (h10 X + h11 Y + h12)/(h20 X + h21 Y + 1)
		A[r+1][0] = 0
		A[r+1][1] = 0
		A[r+1][2] = 0
		A[r+1][3] = X
		A[r+1][4] = Y
		A[r+1][5] = 1
		A[r+1][6] = -X * y
		A[r+1][7] = -Y * y
		b[r+1] = y
	}

	// Solve using Gaussian elimination
	h, ok := solve8x8(A, b)
	if !ok {
		return [9]float64{}, false
	}
	H := [9]float64{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}

	return H, true
}

func solve8x8(a [8][8]float64, b [8]float64) ([8]float64, bool) {
	// Create working copies
	matrix := a
	vector := b

	// Forward elimination with partial pivoting
	for i := range 8 {
		if !pivotAndNormalize(&matrix, &vector, i) {
			return [8]float64{}, false
		}
		eliminateColumn(&matrix, &vector, i)
	}

	// Back substitution
	var x [8]float64
	for i := range 8 {
		x[i] = vector[i]
	}
	return x, true
}

func pivotAndNormalize(matrix *[8][8]float64, vector *[8]float64, col int) bool {
	// Find pivot row
	pivotRow := findPivotRow(*matrix, col)
	if pivotRow == -1 {
		return false
	}

	// Swap rows if needed
	if pivotRow != col {
		swapRows(matrix, vector, col, pivotRow)
	}

	// Normalize pivot row
	normalizeRow(matrix, vector, col)
	return true
}

func findPivotRow(matrix [8][8]float64, col int) int {
	maxAbs := abs(matrix[col][col])
	pivotRow := col
	for r := col + 1; r < 8; r++ {
		if abs(matrix[r][col]) > maxAbs {
			maxAbs = abs(matrix[r][col])
			pivotRow = r
		}
	}
	if maxAbs == 0 {
		return -1
	}
	return pivotRow
}

func swapRows(matrix *[8][8]float64, vector *[8]float64, row1, row2 int) {
	matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
	vector[row1], vector[row2] = vector[row2], vector[row1]
}

func normalizeRow(matrix *[8][8]float64, vector *[8]float64, row int) {
	div := matrix[row][row]
	for c := row; c < 8; c++ {
		matrix[row][c] /= div
	}
	vector[row] /= div
}

func eliminateColumn(matrix *[8][8]float64, vector *[8]float64, col int) {
	for r := range 8 {
		if r == col {
			continue
		}
		factor := matrix[r][col]
		if factor == 0 {
			continue
		}
		for c := col; c < 8; c++ {
			matrix[r][c] -= factor * matrix[col][c]
		}
		vector[r] -= factor * vector[col]
	}
}

func applyHomography(h [9]float64, x, y float64) (float64, float64) {
	denom := h[6]*x + h[7]*y + h[8]
	if denom == 0 {
		return -1e9, -1e9
	}
	sx := (h[0]*x + h[1]*y + h[2]) / denom
	sy := (h[3]*x + h[4]*y + h[5]) / denom
	return sx, sy
}

func bilinearSample(src image.Image, x, y float64) color.Color {
	// Clamp sampling outside bounds to black
	b := src.Bounds()
	if x < float64(b.Min.X) || y < float64(b.Min.Y) || x > float64(b.Max.X-1) || y > float64(b.Max.Y-1) {
		return color.RGBA{0, 0, 0, 255}
	}
	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 >= b.Max.X {
		x1 = b.Max.X - 1
	}
	if y1 >= b.Max.Y {
		y1 = b.Max.Y - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	c00 := toRGBA(src.At(x0, y0))
	c10 := toRGBA(src.At(x1, y0))
	c01 := toRGBA(src.At(x0, y1))
	c11 := toRGBA(src.At(x1, y1))
	r := lerp(lerp(c00.R, c10.R, fx), lerp(c01.R, c11.R, fx), fy)
	g := lerp(lerp(c00.G, c10.G, fx), lerp(c01.G, c11.G, fx), fy)
	bl := lerp(lerp(c00.B, c10.B, fx), lerp(c01.B, c11.B, fx), fy)
	a := lerp(lerp(c00.A, c10.A, fx), lerp(c01.A, c11.A, fx), fy)
	return color.RGBA{uint8(r + 0.5), uint8(g + 0.5), uint8(bl + 0.5), uint8(a + 0.5)}
}

type rgba struct{ R, G, B, A float64 }

func toRGBA(c color.Color) rgba {
	r, g, b, a := c.RGBA()
	return rgba{R: float64(r >> 8), G: float64(g >> 8), B: float64(b >> 8), A: float64(a >> 8)}
}

func lerp(a, b, t float64) float64 { return a + (b-a)*t }
func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func dumpMaskPNG(dir string, mask []float32, w, h int, thr float64) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_mask_%d.png", ts))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		row := y * w
		for x := range w {
			v := float64(mask[row+x])
			// visualize as grayscale, emphasize threshold
			g := uint8(clamp01(v) * 255)
			if v >= thr {
				img.Set(x, y, color.RGBA{R: g, G: 0, B: 0, A: 255}) // red-ish for positive
			} else {
				img.Set(x, y, color.RGBA{R: g, G: g, B: g, A: 255})
			}
		}
	}
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, img)
}

func dumpOverlayPNG(dir string, src image.Image, quad []utils.Point) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_overlay_%d.png", ts))
	// Clone to RGBA and draw polygon
	b := src.Bounds()
	canvas := image.NewRGBA(b)
	// draw original
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			canvas.Set(x, y, src.At(x, y))
		}
	}
	utils.DrawPolygon(canvas, quad, color.RGBA{255, 0, 0, 255}, 2)
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, canvas)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func dumpComparePNG(dir string, src image.Image, srcQuad []utils.Point, dst image.Image) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	ts := time.Now().UnixNano()
	path := filepath.Join(dir, fmt.Sprintf("rect_compare_%d.png", ts))
	sb := src.Bounds()
	db := dst.Bounds()
	gap := 10
	outW := sb.Dx() + gap + db.Dx()
	outH := sb.Dy()
	if db.Dy() > outH {
		outH = db.Dy()
	}
	canvas := image.NewRGBA(image.Rect(0, 0, outW, outH))
	// draw source on left
	for y := range sb.Dy() {
		for x := range sb.Dx() {
			canvas.Set(x, y, src.At(sb.Min.X+x, sb.Min.Y+y))
		}
	}
	// draw destination on right
	xoff := sb.Dx() + gap
	for y := range db.Dy() {
		for x := range db.Dx() {
			canvas.Set(xoff+x, y, dst.At(db.Min.X+x, db.Min.Y+y))
		}
	}
	// overlay quad on left
	utils.DrawPolygon(canvas, srcQuad, color.RGBA{255, 0, 0, 255}, 2)
	// overlay rectangle border on right
	utils.DrawRect(canvas, image.Rect(xoff, 0, xoff+db.Dx(), db.Dy()), color.RGBA{0, 255, 0, 255}, 2)
	f, err := os.Create(path) //nolint:gosec // G304: path is constructed from timestamp in debug directory
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, canvas)
}

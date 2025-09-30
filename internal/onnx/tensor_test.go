package onnx

import "testing"

func TestNewImageTensorAndVerify(t *testing.T) {
	c, h, w := 3, 4, 5
	data := make([]float32, c*h*w)
	ten, err := NewImageTensor(data, c, h, w)
	if err != nil {
		t.Fatalf("NewImageTensor error: %v", err)
	}
	if err := ValidateNCHW(ten.Shape); err != nil {
		t.Fatalf("ValidateNCHW: %v", err)
	}
	if err := VerifyImageTensor(ten); err != nil {
		t.Fatalf("VerifyImageTensor: %v", err)
	}
}

func TestNewImageTensorErrors(t *testing.T) {
	tests := []struct {
		name    string
		data    []float32
		c, h, w int
		wantErr bool
	}{
		{
			name:    "nil data",
			data:    nil,
			c:       3,
			h:       4,
			w:       5,
			wantErr: true,
		},
		{
			name:    "data too short",
			data:    make([]float32, 10),
			c:       3,
			h:       4,
			w:       5,
			wantErr: true,
		},
		{
			name:    "data too long",
			data:    make([]float32, 100),
			c:       3,
			h:       4,
			w:       5,
			wantErr: true,
		},
		{
			name:    "valid data",
			data:    make([]float32, 60),
			c:       3,
			h:       4,
			w:       5,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewImageTensor(tt.data, tt.c, tt.h, tt.w)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewImageTensor() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewBatchImageTensor(t *testing.T) {
	c, h, w := 3, 2, 2
	per := c * h * w
	img1 := make([]float32, per)
	img2 := make([]float32, per)
	ten, err := NewBatchImageTensor([][]float32{img1, img2}, c, h, w)
	if err != nil {
		t.Fatalf("NewBatchImageTensor error: %v", err)
	}
	if len(ten.Shape) != 4 || ten.Shape[0] != 2 || ten.Shape[1] != int64(c) ||
		ten.Shape[2] != int64(h) || ten.Shape[3] != int64(w) {
		t.Fatalf("unexpected shape: %v", ten.Shape)
	}
	if err := VerifyImageTensor(ten); err != nil {
		t.Fatalf("VerifyImageTensor: %v", err)
	}
}

func TestNewBatchImageTensorErrors(t *testing.T) {
	c, h, w := 3, 2, 2
	per := c * h * w

	tests := []struct {
		name    string
		images  [][]float32
		c, h, w int
		wantErr bool
	}{
		{
			name:    "empty batch",
			images:  [][]float32{},
			c:       c,
			h:       h,
			w:       w,
			wantErr: true,
		},
		{
			name:    "nil batch",
			images:  nil,
			c:       c,
			h:       h,
			w:       w,
			wantErr: true,
		},
		{
			name: "mismatched image size - too short",
			images: [][]float32{
				make([]float32, per),
				make([]float32, per-1),
			},
			c:       c,
			h:       h,
			w:       w,
			wantErr: true,
		},
		{
			name: "mismatched image size - too long",
			images: [][]float32{
				make([]float32, per),
				make([]float32, per+1),
			},
			c:       c,
			h:       h,
			w:       w,
			wantErr: true,
		},
		{
			name: "valid batch",
			images: [][]float32{
				make([]float32, per),
				make([]float32, per),
			},
			c:       c,
			h:       h,
			w:       w,
			wantErr: false,
		},
		{
			name: "single image batch",
			images: [][]float32{
				make([]float32, per),
			},
			c:       c,
			h:       h,
			w:       w,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewBatchImageTensor(tt.images, tt.c, tt.h, tt.w)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewBatchImageTensor() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateNCHW(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int64
		wantErr bool
	}{
		{
			name:    "valid NCHW",
			shape:   []int64{1, 3, 224, 224},
			wantErr: false,
		},
		{
			name:    "wrong rank - 2D",
			shape:   []int64{3, 224},
			wantErr: true,
		},
		{
			name:    "wrong rank - 3D",
			shape:   []int64{1, 3, 224},
			wantErr: true,
		},
		{
			name:    "wrong rank - 5D",
			shape:   []int64{1, 3, 224, 224, 1},
			wantErr: true,
		},
		{
			name:    "zero N dimension",
			shape:   []int64{0, 3, 224, 224},
			wantErr: true,
		},
		{
			name:    "zero C dimension",
			shape:   []int64{1, 0, 224, 224},
			wantErr: true,
		},
		{
			name:    "zero H dimension",
			shape:   []int64{1, 3, 0, 224},
			wantErr: true,
		},
		{
			name:    "zero W dimension",
			shape:   []int64{1, 3, 224, 0},
			wantErr: true,
		},
		{
			name:    "negative N dimension",
			shape:   []int64{-1, 3, 224, 224},
			wantErr: true,
		},
		{
			name:    "negative C dimension",
			shape:   []int64{1, -3, 224, 224},
			wantErr: true,
		},
		{
			name:    "negative H dimension",
			shape:   []int64{1, 3, -224, 224},
			wantErr: true,
		},
		{
			name:    "negative W dimension",
			shape:   []int64{1, 3, 224, -224},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateNCHW(tt.shape)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateNCHW() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTensorStats(t *testing.T) {
	tests := []struct {
		name     string
		data     []float32
		wantMin  float32
		wantMax  float32
		wantMean float32
		epsilon  float32
	}{
		{
			name:     "normal values",
			data:     []float32{0, 0.5, 1.0},
			wantMin:  0,
			wantMax:  1.0,
			wantMean: 0.5,
			epsilon:  0.01,
		},
		{
			name:     "empty slice",
			data:     []float32{},
			wantMin:  0,
			wantMax:  0,
			wantMean: 0,
			epsilon:  0.01,
		},
		{
			name:     "single element",
			data:     []float32{42.0},
			wantMin:  42.0,
			wantMax:  42.0,
			wantMean: 42.0,
			epsilon:  0.01,
		},
		{
			name:     "negative values",
			data:     []float32{-1.0, -0.5, 0, 0.5, 1.0},
			wantMin:  -1.0,
			wantMax:  1.0,
			wantMean: 0.0,
			epsilon:  0.01,
		},
		{
			name:     "all same values",
			data:     []float32{5.0, 5.0, 5.0, 5.0},
			wantMin:  5.0,
			wantMax:  5.0,
			wantMean: 5.0,
			epsilon:  0.01,
		},
		{
			name:     "descending values",
			data:     []float32{10.0, 5.0, 2.0, 1.0, 0.0},
			wantMin:  0.0,
			wantMax:  10.0,
			wantMean: 3.6,
			epsilon:  0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			minVal, maxVal, mean := TensorStats(tt.data)
			if minVal != tt.wantMin {
				t.Errorf("TensorStats() min = %v, want %v", minVal, tt.wantMin)
			}
			if maxVal != tt.wantMax {
				t.Errorf("TensorStats() max = %v, want %v", maxVal, tt.wantMax)
			}
			if mean < tt.wantMean-tt.epsilon || mean > tt.wantMean+tt.epsilon {
				t.Errorf("TensorStats() mean = %v, want %v ± %v", mean, tt.wantMean, tt.epsilon)
			}
		})
	}
}

func TestVerifyImageTensor(t *testing.T) {
	tests := []struct {
		name    string
		tensor  Tensor
		wantErr bool
	}{
		{
			name: "valid tensor",
			tensor: Tensor{
				Data:  make([]float32, 60),
				Shape: []int64{1, 3, 4, 5},
			},
			wantErr: false,
		},
		{
			name: "invalid shape - wrong rank",
			tensor: Tensor{
				Data:  make([]float32, 60),
				Shape: []int64{3, 4, 5},
			},
			wantErr: true,
		},
		{
			name: "invalid shape - zero dimension",
			tensor: Tensor{
				Data:  make([]float32, 0),
				Shape: []int64{1, 3, 0, 5},
			},
			wantErr: true,
		},
		{
			name: "mismatched data length - too short",
			tensor: Tensor{
				Data:  make([]float32, 50),
				Shape: []int64{1, 3, 4, 5},
			},
			wantErr: true,
		},
		{
			name: "mismatched data length - too long",
			tensor: Tensor{
				Data:  make([]float32, 70),
				Shape: []int64{1, 3, 4, 5},
			},
			wantErr: true,
		},
		{
			name: "batch tensor",
			tensor: Tensor{
				Data:  make([]float32, 120),
				Shape: []int64{2, 3, 4, 5},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := VerifyImageTensor(tt.tensor)
			if (err != nil) != tt.wantErr {
				t.Errorf("VerifyImageTensor() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

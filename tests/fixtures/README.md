# Test Fixtures

This directory contains test images used for OpenCV.js parity testing.

## Required Test Images

The following images are needed for the test suite:

1. **lenna.png** (512x512) - Classic test image
2. **shapes.png** (640x480) - Geometric shapes for edge detection
3. **text.png** (640x480) - Text rendering for OCR tests
4. **gradient.png** (512x512) - Smooth gradients for blur tests
5. **noise.png** (512x512) - Random noise for filtering tests
6. **edges.png** (512x512) - Sharp edges for edge detection tests

## Quick Setup

Run the setup script to download/generate all test images:

```bash
cd tests/fixtures
python3 generate_fixtures.py
```

Or use the Node.js script:

```bash
cd tests/fixtures
node generate_fixtures.js
```

## Manual Setup

### Option 1: Download from OpenCV samples

```bash
# Lenna image (public domain)
wget https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lenna.png

# Or use a similar test image
wget https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/fruits.jpg -O lenna.png
```

### Option 2: Use any test images

You can use any images for testing. Just make sure they are:
- Named correctly (lenna.png, shapes.png, etc.)
- Reasonable size (512x512 to 1920x1080)
- PNG or JPG format
- License-compatible with open source use

### Option 3: Generate programmatically

See `generate_fixtures.py` or `generate_fixtures.js` for examples of generating
test images programmatically using canvas/PIL.

## License

Test fixtures should be:
- Public domain, OR
- Licensed for testing purposes, OR
- Locally generated (no distribution)

**Note:** We do NOT distribute test images in the git repository to avoid
licensing issues. You must generate or download them locally.

## Verification

After setup, verify all images are present:

```bash
ls -lh tests/fixtures/*.png
```

Expected output:
```
lenna.png (exists)
shapes.png (exists)
text.png (exists)
gradient.png (exists)
noise.png (exists)
edges.png (exists)
```

## Image Specifications

### lenna.png
- Size: 512x512
- Type: Color (RGB/RGBA)
- Content: Portrait with varied textures
- Purpose: General-purpose testing

### shapes.png
- Size: 640x480
- Type: Color or Grayscale
- Content: Circles, rectangles, triangles
- Purpose: Edge detection, contour finding

### text.png
- Size: 640x480
- Type: Grayscale
- Content: Various fonts and sizes
- Purpose: Threshold, adaptive threshold

### gradient.png
- Size: 512x512
- Type: Color
- Content: Smooth color gradients
- Purpose: Blur filters, smoothness

### noise.png
- Size: 512x512
- Type: Color
- Content: Random noise
- Purpose: Noise reduction filters

### edges.png
- Size: 512x512
- Type: Grayscale
- Content: Sharp black/white transitions
- Purpose: Edge detection (Canny, Sobel)

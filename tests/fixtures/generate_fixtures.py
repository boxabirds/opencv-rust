#!/usr/bin/env python3
"""
Generate test fixtures for OpenCV.js parity tests

This script creates synthetic test images when real images
are not available or for consistent reproducibility.

Requirements:
    pip install pillow numpy

Usage:
    python3 generate_fixtures.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_lenna_alternative(size=(512, 512)):
    """
    Create a synthetic test image similar to Lenna
    with varied textures and colors
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Create colorful regions with different textures
    # Top-left: Red gradient
    for y in range(size[1] // 2):
        for x in range(size[0] // 2):
            img[y, x] = [255 - y // 2, 0, 0]

    # Top-right: Green gradient
    for y in range(size[1] // 2):
        for x in range(size[0] // 2, size[0]):
            img[y, x] = [0, 255 - y // 2, 0]

    # Bottom-left: Blue gradient
    for y in range(size[1] // 2, size[1]):
        for x in range(size[0] // 2):
            img[y, x] = [0, 0, 255 - (y - size[1] // 2) // 2]

    # Bottom-right: Mixed
    for y in range(size[1] // 2, size[1]):
        for x in range(size[0] // 2, size[0]):
            img[y, x] = [
                (x - size[0] // 2) // 2,
                (y - size[1] // 2) // 2,
                255 - (x - size[0] // 2) // 2
            ]

    # Add some circular features
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw circles
    for i in range(5):
        x = (i + 1) * size[0] // 6
        y = size[1] // 2
        radius = 30 + i * 10
        color = [(i * 50) % 255, (i * 80) % 255, (i * 100) % 255]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=tuple(color), outline=(255, 255, 255), width=2)

    return pil_img

def create_shapes(size=(640, 480)):
    """Create an image with geometric shapes"""
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Rectangle
    draw.rectangle([50, 50, 200, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=3)

    # Circle
    draw.ellipse([250, 50, 400, 200], fill=(0, 255, 0), outline=(0, 0, 0), width=3)

    # Triangle
    draw.polygon([(500, 50), (450, 150), (550, 150)],
                 fill=(0, 0, 255), outline=(0, 0, 0))

    # Pentagon
    cx, cy, r = 150, 350, 80
    points = [(cx + r * np.cos(2 * np.pi * i / 5), cy + r * np.sin(2 * np.pi * i / 5))
              for i in range(5)]
    draw.polygon(points, fill=(255, 255, 0), outline=(0, 0, 0), width=3)

    # Hexagon
    cx, cy, r = 400, 350, 80
    points = [(cx + r * np.cos(2 * np.pi * i / 6), cy + r * np.sin(2 * np.pi * i / 6))
              for i in range(6)]
    draw.polygon(points, fill=(255, 0, 255), outline=(0, 0, 0), width=3)

    return img

def create_text(size=(640, 480)):
    """Create an image with text"""
    img = Image.new('L', size, color=255)  # Grayscale
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to bitmap font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw text at various sizes
    draw.text((50, 50), "OpenCV Test", fill=0, font=font_large)
    draw.text((50, 150), "Threshold & Detection", fill=0, font=font_medium)
    draw.text((50, 250), "1234567890", fill=0, font=font_medium)
    draw.text((50, 350), "abcdefghijklmnopqrstuvwxyz", fill=0, font=font_small)

    return img

def create_gradient(size=(512, 512)):
    """Create smooth gradients"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    for y in range(size[1]):
        for x in range(size[0]):
            # Radial gradient
            dx = x - size[0] / 2
            dy = y - size[1] / 2
            dist = np.sqrt(dx * dx + dy * dy)
            max_dist = np.sqrt((size[0] / 2) ** 2 + (size[1] / 2) ** 2)

            intensity = int(255 * (1 - dist / max_dist))
            img[y, x] = [
                intensity,
                int(255 * x / size[0]),
                int(255 * y / size[1])
            ]

    return Image.fromarray(img)

def create_noise(size=(512, 512)):
    """Create random noise"""
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(noise)

def create_edges(size=(512, 512)):
    """Create sharp edges for edge detection testing"""
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)

    # Horizontal lines
    for i in range(5):
        y = (i + 1) * size[1] // 6
        draw.line([(0, y), (size[0], y)], fill=0, width=5)

    # Vertical lines
    for i in range(5):
        x = (i + 1) * size[0] // 6
        draw.line([(x, 0), (x, size[1])], fill=0, width=5)

    # Diagonal
    draw.line([(0, 0), (size[0], size[1])], fill=0, width=7)
    draw.line([(size[0], 0), (0, size[1])], fill=0, width=7)

    # Circles
    for i in range(3):
        r = (i + 1) * min(size) // 8
        cx, cy = size[0] // 2, size[1] // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=5)

    return img

def main():
    """Generate all test fixtures"""
    print("Generating test fixtures...")

    fixtures = {
        'lenna.png': create_lenna_alternative,
        'shapes.png': create_shapes,
        'text.png': create_text,
        'gradient.png': create_gradient,
        'noise.png': create_noise,
        'edges.png': create_edges,
    }

    for filename, generator in fixtures.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)

        if os.path.exists(filepath):
            print(f"  ⚠ {filename} already exists, skipping")
            continue

        print(f"  Creating {filename}...")
        img = generator()
        img.save(filepath)
        print(f"  ✓ Saved {filepath}")

    print("\n✓ Test fixtures generated successfully!")
    print("\nVerify with: ls -lh tests/fixtures/*.png")

if __name__ == '__main__':
    main()

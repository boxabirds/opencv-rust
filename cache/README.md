# Cached Dependencies

This directory contains cached copies of external dependencies to avoid network/CDN issues during testing.

## opencv.js

**Version:** 4.8.0
**Source:** https://docs.opencv.org/4.8.0/opencv.js
**Size:** 9.6 MB
**Date Downloaded:** 2025-11-12
**License:** Apache 2.0 (same as OpenCV)

### Why Cached?

The OpenCV.js parity tests require loading opencv.js as a reference implementation. Loading from CDN causes issues:

1. **Container environments** - Network restrictions prevent CDN access
2. **CI reliability** - CDN outages break tests
3. **Reproducibility** - Specific version ensures consistent results
4. **Offline development** - Tests work without internet

### Usage

The test harness automatically loads from `/cache/opencv.js`:

```html
<script async src="/cache/opencv.js"></script>
```

### Updating

To update to a newer version:

```bash
cd cache
wget https://docs.opencv.org/4.10.0/opencv.js -O opencv.js
```

Update the version info in this README.

### Gitignore

This file is **checked into git** despite its size (9.6 MB) because:
- Tests must work in all environments (CI, containers, offline)
- One-time cost, rarely updated
- Critical for parity testing infrastructure

If size becomes an issue, alternatives:
- Use Git LFS
- Download in CI as a build step
- Host on project CDN

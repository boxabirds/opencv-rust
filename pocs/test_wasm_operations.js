/**
 * Test which WASM operations are actually implemented vs stubs
 */
import { chromium } from '@playwright/test';

const OPERATIONS_TO_TEST = [
    // Filters
    { name: 'gaussianBlur', args: [5, 1.5], needsRgba: true },
    { name: 'boxBlur', args: [5], needsRgba: true },
    { name: 'medianBlur', args: [5], needsRgba: true },
    { name: 'bilateralFilter', args: [9, 75, 75], needsRgba: true },
    { name: 'guidedFilter', args: [5, 0.1], needsRgba: true },
    { name: 'gaborFilter', args: [0.1, 0, 3.0], needsRgba: true },
    { name: 'logFilter', args: [5, 1.0], needsRgba: true },
    { name: 'nlmDenoising', args: [10, 7, 21], needsRgba: true },
    { name: 'anisotropicDiffusion', args: [10, 50, 0.1], needsRgba: true },
    { name: 'distanceTransform', args: [2, 3], needsGray: true },
    { name: 'watershed', args: [], needsGray: true },

    // Edges
    { name: 'canny', args: [50, 150, 3], needsGray: true },
    { name: 'sobel', args: [1, 0, 3], needsGray: true },
    { name: 'scharr', args: [1, 0], needsGray: true },
    { name: 'laplacian', args: [3], needsGray: true },

    // Transforms
    { name: 'resize', args: [128, 128], needsRgba: true },
    { name: 'flip', args: [1], needsRgba: true },
    { name: 'rotate', args: [0], needsRgba: true },
    { name: 'warpAffine', args: [[1, 0, 0, 0, 1, 0]], needsRgba: true },
    { name: 'warpPerspective', args: [[1, 0, 0, 0, 1, 0, 0, 0, 1]], needsRgba: true },
    { name: 'getRotationMatrix2D', args: [[128, 128], 45, 1.0], needsRgba: false },

    // Color
    { name: 'cvtColorGray', args: [], needsRgba: true },
    { name: 'cvtColorHsv', args: [], needsRgba: true },
    { name: 'cvtColorLab', args: [], needsRgba: true },
    { name: 'cvtColorYCrCb', args: [], needsRgba: true },
    { name: 'threshold', args: [127, 255, 0], needsGray: true },
    { name: 'adaptiveThreshold', args: [255, 1, 0, 11, 2], needsGray: true },

    // Histogram
    { name: 'calcHistogram', args: [256, [0, 255]], needsGray: true },
    { name: 'equalizeHistogram', args: [], needsGray: true },
    { name: 'normalizeHistogram', args: [1.0, 0.0], needsGray: true },
    { name: 'compareHistograms', args: [0], needsGray: false },
    { name: 'backProjection', args: [[0, 1]], needsRgba: true },

    // Morphology
    { name: 'erode', args: [5, 0, 1], needsGray: true },
    { name: 'dilate', args: [5, 0, 1], needsGray: true },
    { name: 'morphologyOpening', args: [5, 0], needsGray: true },
    { name: 'morphologyClosing', args: [5, 0], needsGray: true },
    { name: 'morphologyGradient', args: [5], needsGray: true },
    { name: 'morphologyTopHat', args: [9], needsGray: true },
    { name: 'morphologyBlackHat', args: [9], needsGray: true },

    // Contours
    { name: 'findContours', args: [0, 1], needsGray: true },
    { name: 'approxPolyDP', args: [2.0, true], needsContour: true },
    { name: 'contourArea', args: [], needsContour: true },
    { name: 'arcLength', args: [true], needsContour: true },
    { name: 'boundingRect', args: [], needsContour: true },
    { name: 'moments', args: [], needsGray: true },

    // Features
    { name: 'harrisCorners', args: [3, 3, 0.04, 0.01], needsGray: true },
    { name: 'goodFeaturesToTrack', args: [100, 0.01, 10], needsGray: true },
    { name: 'fast', args: [10, true], needsGray: true },
    { name: 'sift', args: [0, 3, 0.04], needsGray: true },
    { name: 'orb', args: [500, 1.2, 8], needsGray: true },
    { name: 'brisk', args: [30, 3], needsGray: true },
    { name: 'akaze', args: [0.001, 4], needsGray: true },
    { name: 'kaze', args: [0.001, 4], needsGray: true },
    { name: 'bruteForceMatcher', args: [0, false], needsDescriptors: true },

    // Hough
    { name: 'houghLines', args: [1, 1, 150], needsGray: true },
    { name: 'houghLinesP', args: [1, 1, 80, 30, 10], needsGray: true },
    { name: 'houghCircles', args: [1, 50, 100, 30, 10, 100], needsGray: true },
];

(async () => {
    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
        headless: true
    });
    const page = await browser.newPage();

    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    // Create a simple test HTML page
    await page.setContent(`
        <!DOCTYPE html>
        <html>
        <head><title>WASM Operation Test</title></head>
        <body>
            <div id="status">Loading...</div>
            <script type="module">
                window.testResults = [];
                window.testComplete = false;

                const wasmModule = await import('/pocs/pkg/opencv_rust.js');
                await wasmModule.default();

                // Create test image data
                const size = 256;
                const rgbaData = new Uint8Array(size * size * 4);
                const grayData = new Uint8Array(size * size);

                for (let i = 0; i < size * size; i++) {
                    const value = (i * 17 + 128) % 256;
                    grayData[i] = value;
                    rgbaData[i * 4 + 0] = value;
                    rgbaData[i * 4 + 1] = value;
                    rgbaData[i * 4 + 2] = value;
                    rgbaData[i * 4 + 3] = 255;
                }

                const rgbaMat = wasmModule.WasmMat.fromImageData(rgbaData, size, size, 4);
                const grayMat = wasmModule.WasmMat.fromImageData(grayData, size, size, 1);

                window.wasmModule = wasmModule;
                window.rgbaMat = rgbaMat;
                window.grayMat = grayMat;

                document.getElementById('status').textContent = 'Ready';
            </script>
        </body>
        </html>
    `);

    await page.goto('http://localhost:8000/pocs/');
    await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready', { timeout: 10000 });

    console.log('Testing WASM operations...\n');
    console.log('Operation'.padEnd(30) + ' Status'.padEnd(15) + ' Error');
    console.log('='.repeat(80));

    const results = { working: [], notImplemented: [], error: [] };

    for (const op of OPERATIONS_TO_TEST) {
        try {
            const result = await page.evaluate(async ({ name, args, needsRgba, needsGray, needsContour, needsDescriptors }) => {
                try {
                    const mat = needsGray ? window.grayMat : window.rgbaMat;

                    // Skip operations that need special input types for now
                    if (needsContour || needsDescriptors) {
                        return { status: 'skipped', error: 'Needs special input type' };
                    }

                    const fn = window.wasmModule[name];
                    if (!fn) {
                        return { status: 'missing', error: 'Function not exported' };
                    }

                    const res = await fn(mat, ...args);
                    if (res && res.free) {
                        res.free();
                    }

                    return { status: 'ok', error: null };
                } catch (e) {
                    return { status: 'error', error: e.message };
                }
            }, op);

            const statusStr = result.status === 'ok' ? '✓ WORKING' :
                             result.status === 'skipped' ? '⊘ SKIPPED' :
                             result.status === 'missing' ? '✗ MISSING' : '✗ ERROR';

            console.log(
                op.name.padEnd(30) +
                statusStr.padEnd(15) +
                (result.error || '')
            );

            if (result.status === 'ok') {
                results.working.push(op.name);
            } else if (result.error && result.error.includes('not implemented')) {
                results.notImplemented.push(op.name);
            } else if (result.status === 'error') {
                results.error.push({ name: op.name, error: result.error });
            }

        } catch (e) {
            console.log(op.name.padEnd(30) + '✗ ERROR'.padEnd(15) + e.message);
            results.error.push({ name: op.name, error: e.message });
        }
    }

    console.log('\n' + '='.repeat(80));
    console.log('\nSummary:');
    console.log(`Working: ${results.working.length}`);
    console.log(`Not Implemented: ${results.notImplemented.length}`);
    console.log(`Errors: ${results.error.length}`);

    if (results.notImplemented.length > 0) {
        console.log('\nNeed Implementation:');
        results.notImplemented.forEach(name => console.log('  - ' + name));
    }

    if (results.error.length > 0) {
        console.log('\nErrors (need investigation):');
        results.error.slice(0, 10).forEach(({ name, error }) => {
            console.log(`  - ${name}: ${error}`);
        });
    }

    await browser.close();
})();

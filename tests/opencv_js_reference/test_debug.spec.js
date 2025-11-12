import { test, expect } from '@playwright/test';

/**
 * Debug test to capture console output and diagnose initialization issues
 */

test.describe('Test Harness Debug', () => {
  test('comprehensive initialization diagnostics', async ({ page }) => {
    const consoleMessages = [];
    const errors = [];
    const warnings = [];
    const networkRequests = [];
    const networkResponses = [];

    // Capture all console messages by type
    page.on('console', msg => {
      const timestamp = new Date().toISOString();
      const text = msg.text();
      const type = msg.type();

      const entry = `[${timestamp}] [${type.toUpperCase()}] ${text}`;

      consoleMessages.push(entry);

      if (type === 'error') {
        errors.push(entry);
        console.error(entry);
      } else if (type === 'warning') {
        warnings.push(entry);
        console.warn(entry);
      } else {
        console.log(entry);
      }
    });

    // Capture page errors
    page.on('pageerror', error => {
      const timestamp = new Date().toISOString();
      const text = `[${timestamp}] [PAGE ERROR] ${error.name}: ${error.message}\nStack: ${error.stack}`;
      errors.push(text);
      console.error(text);
    });

    // Capture network requests
    page.on('request', request => {
      const timestamp = new Date().toISOString();
      const entry = {
        timestamp,
        method: request.method(),
        url: request.url(),
        resourceType: request.resourceType(),
      };
      networkRequests.push(entry);
      console.log(`[${timestamp}] [REQUEST] ${request.method()} ${request.url()}`);
    });

    // Capture network responses
    page.on('response', response => {
      const timestamp = new Date().toISOString();
      const entry = {
        timestamp,
        status: response.status(),
        url: response.url(),
        size: response.headers()['content-length'] || 'unknown',
      };
      networkResponses.push(entry);
      console.log(`[${timestamp}] [RESPONSE] ${response.status()} ${response.url()} (${entry.size} bytes)`);
    });

    // Capture request failures
    page.on('requestfailed', request => {
      const timestamp = new Date().toISOString();
      const text = `[${timestamp}] [REQUEST FAILED] ${request.url()}: ${request.failure()?.errorText}`;
      errors.push(text);
      console.error(text);
    });

    // Navigate to test harness
    console.log('\n========================================');
    console.log('=== STARTING TEST HARNESS DIAGNOSTICS ===');
    console.log('========================================\n');

    console.log('Step 1: Navigating to test-harness.html...');
    const startTime = Date.now();

    try {
      const response = await page.goto('/tests/opencv_js_reference/test-harness.html', {
        waitUntil: 'domcontentloaded',
        timeout: 30000
      });
      console.log(`Response status: ${response.status()}`);
      console.log(`Response headers:`, response.headers());
    } catch (e) {
      console.error(`Navigation failed: ${e.message}`);
    }

    // Wait and check state periodically
    const checkInterval = 2000; // Check every 2 seconds
    const maxWaitTime = 20000; // Wait up to 20 seconds

    console.log('\nStep 2: Monitoring initialization progress...\n');

    for (let elapsed = 0; elapsed < maxWaitTime; elapsed += checkInterval) {
      await page.waitForTimeout(checkInterval);

      const state = await page.evaluate(() => {
        return {
          testHarnessReady: window.testHarnessReady || false,
          opencv_js_ready: window.opencv_js_ready || false,
          opencv_rust_ready: window.opencv_rust_ready || false,
          testHarnessError: window.testHarnessError ? {
            message: window.testHarnessError.message,
            stack: window.testHarnessError.stack,
            name: window.testHarnessError.name
          } : null,
          cvExists: typeof cv !== 'undefined',
          opencvRustExists: typeof window.opencvRust !== 'undefined',
          documentReadyState: document.readyState,
          statusElement: document.getElementById('status')?.textContent,
          initSteps: window.initSteps || [],
        };
      });

      console.log(`[${elapsed + checkInterval}ms] State check:`, {
        ready: state.testHarnessReady,
        opencv_js: state.opencv_js_ready,
        opencv_rust: state.opencv_rust_ready,
        error: state.testHarnessError?.message || 'none',
        status: state.statusElement
      });

      // If we hit an error, stop early
      if (state.testHarnessError) {
        console.error('\n!!! INITIALIZATION ERROR DETECTED !!!');
        console.error('Error details:', JSON.stringify(state.testHarnessError, null, 2));
        console.error('Init steps before error:', JSON.stringify(state.initSteps, null, 2));
        break;
      }

      // If initialization is complete, stop early
      if (state.testHarnessReady) {
        console.log('\n✓ INITIALIZATION COMPLETE');
        break;
      }
    }

    // Final state check with debug log extraction
    const finalState = await page.evaluate(() => {
      return {
        testHarnessReady: window.testHarnessReady || false,
        opencv_js_ready: window.opencv_js_ready || false,
        opencv_rust_ready: window.opencv_rust_ready || false,
        testHarnessError: window.testHarnessError ? {
          message: window.testHarnessError.message,
          stack: window.testHarnessError.stack,
          name: window.testHarnessError.name
        } : null,
        cvExists: typeof cv !== 'undefined',
        cvMatExists: typeof cv !== 'undefined' && typeof cv.Mat !== 'undefined',
        opencvRustExists: typeof window.opencvRust !== 'undefined',
        documentReadyState: document.readyState,
        statusElement: document.getElementById('status')?.textContent,
        detailsElement: document.getElementById('details')?.textContent,
        initSteps: window.initSteps || [],
        debugLog: window.debugLog || [],
        webGpuAvailable: !!navigator.gpu,
        userAgent: navigator.userAgent,
      };
    });

    // Take screenshot
    await page.screenshot({ path: 'test-harness-debug.png', fullPage: true });

    // Print summary report
    console.log('\n========================================');
    console.log('=== DIAGNOSTIC SUMMARY ===');
    console.log('========================================\n');

    console.log('Final State:');
    console.log(JSON.stringify(finalState, null, 2));

    console.log('\n--- Network Activity ---');
    console.log(`Total requests: ${networkRequests.length}`);
    console.log(`Total responses: ${networkResponses.length}`);
    console.log('\nKey resources:');
    networkResponses.forEach(resp => {
      if (resp.url.includes('opencv') || resp.url.includes('pkg/')) {
        console.log(`  ${resp.status} ${resp.url.split('/').pop()} (${resp.size} bytes)`);
      }
    });

    console.log('\n--- Console Activity ---');
    console.log(`Total console messages: ${consoleMessages.length}`);
    console.log(`Errors: ${errors.length}`);
    console.log(`Warnings: ${warnings.length}`);

    if (errors.length > 0) {
      console.log('\nErrors captured:');
      errors.forEach(err => console.error(err));
    }

    if (warnings.length > 0) {
      console.log('\nWarnings captured:');
      warnings.forEach(warn => console.warn(warn));
    }

    console.log('\n--- Initialization Steps ---');
    if (finalState.initSteps.length > 0) {
      finalState.initSteps.forEach(step => {
        console.log(`  [${step.time.toFixed(2)}ms] ${step.message} (${step.className})`);
      });
    } else {
      console.log('  No init steps recorded');
    }

    console.log('\n--- Debug Log (window.debugLog) ---');
    if (finalState.debugLog && finalState.debugLog.length > 0) {
      console.log(`Total log entries: ${finalState.debugLog.length}`);
      console.log('\nDetailed log:');
      finalState.debugLog.forEach(entry => {
        const level = entry.level.toUpperCase();
        const timeStr = `[${entry.time.toFixed(2)}ms]`;
        const prefix = entry.level === 'error' ? '❌' : entry.level === 'warn' ? '⚠️' : '📝';
        console.log(`${prefix} ${timeStr} [${entry.category}] ${entry.message}`);
      });
    } else {
      console.log('  No debug log entries found');
    }

    console.log('\n--- Files ---');
    console.log('Screenshot saved: test-harness-debug.png');

    const totalTime = Date.now() - startTime;
    console.log(`\nTotal test duration: ${totalTime}ms`);

    console.log('\n========================================');
    console.log('=== END DIAGNOSTICS ===');
    console.log('========================================\n');

    // Test always passes - this is for debugging only
    expect(true).toBe(true);
  });

  test('isolated OpenCV.js load test', async ({ page }) => {
    console.log('\n=== Testing OpenCV.js load in isolation ===');

    const consoleMessages = [];
    page.on('console', msg => consoleMessages.push(`[${msg.type()}] ${msg.text()}`));

    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    // Create minimal HTML that only loads OpenCV.js
    await page.setContent(`
      <!DOCTYPE html>
      <html>
        <head><title>OpenCV.js Only Test</title></head>
        <body>
          <h1>OpenCV.js Only</h1>
          <div id="status">Loading...</div>
          <script src="/cache/opencv.js"></script>
          <script>
            window.onOpenCVReady = () => {
              console.log('✓ OpenCV.js loaded successfully');
              document.getElementById('status').textContent = '✓ Ready';
            };

            setTimeout(() => {
              if (typeof cv === 'undefined') {
                console.error('✗ OpenCV.js did not load in 10 seconds');
              }
            }, 10000);
          </script>
        </body>
      </html>
    `);

    await page.waitForTimeout(12000);

    const cvLoaded = await page.evaluate(() => typeof cv !== 'undefined');

    console.log('OpenCV.js loaded:', cvLoaded);
    console.log('Console messages:', consoleMessages);
    if (errors.length > 0) {
      console.error('Errors:', errors);
    }

    expect(true).toBe(true);
  });

  test('isolated WASM load test', async ({ page }) => {
    console.log('\n=== Testing WASM load in isolation ===');

    const consoleMessages = [];
    page.on('console', msg => consoleMessages.push(`[${msg.type()}] ${msg.text()}`));

    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    // Create minimal HTML that only loads our WASM
    await page.setContent(`
      <!DOCTYPE html>
      <html>
        <head><title>WASM Only Test</title></head>
        <body>
          <h1>WASM Only</h1>
          <div id="status">Loading...</div>
          <script type="module">
            console.log('Starting WASM load...');

            try {
              const module = await import('/pkg/opencv_rust.js');
              console.log('Module imported:', Object.keys(module));

              await module.default();
              console.log('✓ WASM initialized');

              window.opencvRust = module;
              document.getElementById('status').textContent = '✓ Ready';
            } catch (error) {
              console.error('✗ WASM load failed:', error.message);
              console.error('Stack:', error.stack);
              document.getElementById('status').textContent = '✗ Error: ' + error.message;
            }
          </script>
        </body>
      </html>
    `);

    await page.waitForTimeout(10000);

    const wasmLoaded = await page.evaluate(() => typeof window.opencvRust !== 'undefined');
    const status = await page.evaluate(() => document.getElementById('status').textContent);

    console.log('WASM loaded:', wasmLoaded);
    console.log('Status:', status);
    console.log('Console messages:', consoleMessages);
    if (errors.length > 0) {
      console.error('Errors:', errors);
    }

    expect(true).toBe(true);
  });
});

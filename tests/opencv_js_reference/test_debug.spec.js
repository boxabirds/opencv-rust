import { test, expect } from '@playwright/test';

/**
 * Debug test to capture console output and diagnose initialization issues
 */

test.describe('Test Harness Debug', () => {
  test('capture initialization console output', async ({ page }) => {
    const consoleMessages = [];
    const errors = [];

    // Capture all console messages
    page.on('console', msg => {
      const text = `[${msg.type()}] ${msg.text()}`;
      consoleMessages.push(text);
      console.log(text);
    });

    // Capture page errors
    page.on('pageerror', error => {
      const text = `[PAGE ERROR] ${error.message}\n${error.stack}`;
      errors.push(text);
      console.error(text);
    });

    // Navigate to test harness
    console.log('\n=== Navigating to test harness ===');
    const response = await page.goto('/tests/opencv_js_reference/test-harness.html');
    console.log(`Response status: ${response.status()}`);

    // Wait a bit to see what happens
    console.log('\n=== Waiting 10 seconds for initialization ===');
    await page.waitForTimeout(10000);

    // Check current state
    const state = await page.evaluate(() => {
      return {
        testHarnessReady: window.testHarnessReady || false,
        opencv_js_ready: window.opencv_js_ready || false,
        opencv_rust_ready: window.opencv_rust_ready || false,
        testHarnessError: window.testHarnessError ? window.testHarnessError.message : null,
        cvExists: typeof cv !== 'undefined',
        opencvRustExists: typeof window.opencvRust !== 'undefined',
        documentReadyState: document.readyState,
        statusElement: document.getElementById('status')?.textContent,
      };
    });

    console.log('\n=== Current State ===');
    console.log(JSON.stringify(state, null, 2));

    console.log('\n=== Console Messages ===');
    consoleMessages.forEach(msg => console.log(msg));

    if (errors.length > 0) {
      console.log('\n=== Page Errors ===');
      errors.forEach(err => console.error(err));
    }

    // Take screenshot
    await page.screenshot({ path: 'test-harness-debug.png', fullPage: true });
    console.log('\n=== Screenshot saved to test-harness-debug.png ===');

    // Check if WASM file is accessible
    console.log('\n=== Checking /pkg/opencv_rust.js accessibility ===');
    const wasmResponse = await page.goto('/pkg/opencv_rust.js');
    console.log(`WASM file status: ${wasmResponse.status()}`);

    // This test always passes - it's just for debugging
    expect(true).toBe(true);
  });

  test('minimal initialization check', async ({ page }) => {
    // Even simpler test - just check if page loads
    const response = await page.goto('/tests/opencv_js_reference/test-harness.html');
    expect(response.status()).toBe(200);

    // Check if HTML rendered
    const title = await page.title();
    console.log(`Page title: ${title}`);
    expect(title).toContain('OpenCV.js Parity Test Harness');
  });
});

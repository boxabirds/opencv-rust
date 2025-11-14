// Minimal test for nlmDenoising function
import { chromium } from 'playwright';

async function runTest() {
  console.log('Starting minimal nlmDenoising test...');
  const startTime = Date.now();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Set a 10 second timeout for the entire test
  page.setDefaultTimeout(10000);

  try {
    // Load the simple test HTML page from HTTP server
    const testUrl = 'http://localhost:8888/test-nlm-simple.html';
    console.log('Loading test page:', testUrl);

    // Listen for console messages
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));

    await page.goto(testUrl);
    console.log('Page loaded');

    // Wait for test to complete (max 10 seconds)
    await page.waitForTimeout(10000);

    // Get the output text
    const output = await page.evaluate(() => {
      return document.getElementById('output').innerText;
    });

    console.log('\n=== TEST OUTPUT ===');
    console.log(output);
    console.log('===================\n');

    // Check if test passed
    const passed = output.includes('✓ TEST PASSED');
    const failed = output.includes('✗ TEST FAILED');
    const timeout = output.includes('TIMEOUT');

    await browser.close();

    const totalTime = Date.now() - startTime;
    console.log('Total execution time:', totalTime, 'ms');

    if (passed) {
      console.log('\n✓ nlmDenoising test PASSED');
      process.exit(0);
    } else if (timeout) {
      console.log('\n✗ nlmDenoising test TIMED OUT (function hung)');
      process.exit(1);
    } else if (failed) {
      console.log('\n✗ nlmDenoising test FAILED');
      process.exit(1);
    } else {
      console.log('\n? Test status unclear');
      process.exit(1);
    }

  } catch (error) {
    console.error('Test error:', error.message);
    console.error(error.stack);
    await browser.close();
    process.exit(1);
  }
}

runTest();

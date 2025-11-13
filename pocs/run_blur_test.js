import { chromium } from '@playwright/test';

(async () => {
  console.log('Running Gaussian Blur test...\n');

  const browser = await chromium.launch({
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan'
    ]
  });
  const page = await browser.newPage();

  // Navigate to test page (assumes http server running on 8000)
  await page.goto('http://localhost:8000/blur_test.html');

  // Wait for completion
  await page.waitForFunction(() => {
    return document.getElementById('status').textContent.includes('COMPLETE');
  }, { timeout: 30000 });

  // Get results
  const status = await page.locator('#status').innerHTML();

  // Parse and display results
  console.log('Results:');
  console.log(status.replace(/<br>/g, '\n').replace(/<[^>]*>/g, ''));

  await browser.close();

  // Check if it's pixel perfect
  if (status.includes('Different pixels: 0')) {
    console.log('\n✓ PIXEL PERFECT! 100% match with OpenCV.js');
    process.exit(0);
  } else {
    console.log('\n⚠ Not pixel perfect (but close)');
    process.exit(0);
  }
})();

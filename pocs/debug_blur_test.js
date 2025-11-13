import { chromium } from '@playwright/test';

(async () => {
  console.log('Running Gaussian Blur test with debug logging...\n');

  const browser = await chromium.launch({
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan'
    ]
  });
  const page = await browser.newPage();

  // Listen to console logs
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  // Navigate to test page
  await page.goto('http://localhost:8000/blur_test.html');

  // Check status every 2 seconds
  for (let i = 0; i < 15; i++) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    const status = await page.locator('#status').textContent();
    console.log(`[${i*2}s] Status:`, status);

    if (status.includes('COMPLETE') || status.includes('Error:')) {
      break;
    }
  }

  // Get final results
  const status = await page.locator('#status').innerHTML();
  console.log('\nFinal result:');
  console.log(status.replace(/<br>/g, '\n').replace(/<[^>]*>/g, ''));

  await browser.close();
})();

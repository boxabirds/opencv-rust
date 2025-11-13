import { chromium } from '@playwright/test';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

(async () => {
  console.log('Starting WebGPU test...');

  const browser = await chromium.launch({
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan'
    ]
  });
  const page = await browser.newPage();

  // Load local HTML file
  await page.goto(`file://${__dirname}/hello.html`);

  // Wait for status to update
  await page.waitForFunction(() => {
    const status = document.getElementById('status').textContent;
    return status !== 'Checking...';
  }, { timeout: 5000 });

  // Get results
  const status = await page.locator('#status').textContent();
  const result = await page.locator('#result').textContent();

  console.log('\nStatus:', status);
  console.log('Result:', result);

  await browser.close();

  if (status.includes('available')) {
    console.log('\n✓ WebGPU is working!');
    process.exit(0);
  } else {
    console.log('\n✗ WebGPU test failed');
    process.exit(1);
  }
})();

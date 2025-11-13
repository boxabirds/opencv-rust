// Script to save Gaussian blur comparison images
// Run with: node save_comparison.js

import { chromium } from '@playwright/test';
import fs from 'fs';
import path from 'path';

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  // Go to debug page
  await page.goto('http://localhost:8787/debug_gaussian.html');

  console.log('Waiting for processing to complete...');
  await page.waitForFunction(() => {
    const status = document.getElementById('status').textContent;
    return status.includes('Processing complete');
  }, { timeout: 30000 });

  console.log('Saving images...');
  const outputDir = './comparison_output';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }

  // Save each canvas as PNG
  const canvases = ['original', 'opencv', 'ours', 'diff'];
  for (const canvasId of canvases) {
    const dataUrl = await page.evaluate((id) => {
      return document.getElementById(id).toDataURL('image/png');
    }, canvasId);

    const base64Data = dataUrl.replace(/^data:image\/png;base64,/, '');
    const filePath = path.join(outputDir, `gaussian_${canvasId}.png`);
    fs.writeFileSync(filePath, base64Data, 'base64');
    console.log(`  ✓ Saved ${filePath}`);
  }

  // Get and save statistics
  const stats = await page.evaluate(() => {
    return document.getElementById('stats').textContent;
  });
  fs.writeFileSync(path.join(outputDir, 'statistics.txt'), stats);
  console.log(`  ✓ Saved statistics.txt`);

  await browser.close();
  console.log('\nAll images saved to ./comparison_output/');
  console.log('You can now visually compare:');
  console.log('  - gaussian_opencv.png (reference)');
  console.log('  - gaussian_ours.png (our implementation)');
  console.log('  - gaussian_diff.png (10x amplified difference)');
})();

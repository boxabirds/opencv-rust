// Script to run the pixel difference diagnostic
// Run with: node run_diagnostic.js

import { chromium } from '@playwright/test';

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  // Go to diagnostic page
  await page.goto('http://localhost:8787/diagnose_diff.html');

  console.log('Waiting for analysis to complete...');
  await page.waitForFunction(() => {
    const status = document.getElementById('status').textContent;
    return status.includes('Analysis complete');
  }, { timeout: 30000 });

  console.log('\nCapturing results...\n');

  // Get the output HTML and extract text
  const output = await page.evaluate(() => {
    return document.getElementById('output').innerText;
  });

  console.log(output);

  await browser.close();
})();

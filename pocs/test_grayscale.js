import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
    const page = await browser.newPage();
    await page.goto('http://localhost:8000/pocs/test_grayscale_debug.html');
    await page.waitForTimeout(5000);
    const results = await page.evaluate(() => document.getElementById('results').innerText);
    console.log(results);
    await browser.close();
})();

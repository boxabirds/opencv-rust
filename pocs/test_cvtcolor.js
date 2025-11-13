import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
    const page = await browser.newPage();

    page.on('console', msg => console.log('PAGE:', msg.text()));

    await page.goto('http://localhost:8000/pocs/test_cvtcolor_debug.html');
    await page.waitForFunction(() =>
        document.getElementById('status').textContent.includes('Complete') ||
        document.getElementById('status').textContent.includes('Error'),
        { timeout: 15000 }
    );
    const status = await page.evaluate(() => document.getElementById('status').textContent);
    const result = await page.evaluate(() => document.getElementById('result').textContent);
    console.log('\n' + status);
    console.log(result);
    await browser.close();
})();

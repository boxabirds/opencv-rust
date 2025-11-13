import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
    const page = await browser.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('ERROR') || text.includes('Warning')) {
            console.log('PAGE:', text);
        }
    });

    await page.goto('http://localhost:8000/pocs/test_sobel_comparison.html');
    await page.waitForFunction(() =>
        document.getElementById('status').textContent.includes('Complete') ||
        document.getElementById('status').textContent.includes('Error'),
        { timeout: 60000 }
    );

    const results = await page.evaluate(() => document.getElementById('results').innerText);
    console.log(results);

    await browser.close();
})();

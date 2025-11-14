import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
        headless: true
    });
    const page = await browser.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (!text.includes('WebGL') && !text.includes('DevTools')) {
            console.log('PAGE:', text);
        }
    });
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    await page.goto('http://localhost:8000/pocs/test_wasm_operations.html');

    // Wait for test to complete
    await page.waitForFunction(() => {
        const status = document.getElementById('status').textContent;
        return status.includes('complete') || status.includes('ERROR');
    }, { timeout: 60000 });

    const status = await page.$eval('#status', el => el.textContent);
    const results = await page.$eval('#results', el => el.textContent);

    console.log(results);

    await browser.close();
})();

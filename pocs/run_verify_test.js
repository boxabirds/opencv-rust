import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
    });
    const page = await browser.newPage();

    // Capture console messages
    page.on('console', msg => console.log('PAGE:', msg.text()));

    // Capture errors
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    await page.goto('http://localhost:8000/pocs/verify_rgba_fix.html');

    // Wait for tests to complete or error
    await page.waitForFunction(() => {
        const status = document.getElementById('status').textContent;
        return status.includes('complete') || status.includes('Error');
    }, { timeout: 30000 });

    // Get results
    const status = await page.$eval('#status', el => el.textContent);
    const results = await page.$eval('#results', el => el.textContent);

    console.log('\n' + '='.repeat(80));
    console.log('RGBA Fix Verification Results');
    console.log('='.repeat(80));
    console.log('Status:', status);
    console.log('\n' + results);
    console.log('='.repeat(80));

    await browser.close();
})();

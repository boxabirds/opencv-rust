import { chromium } from '@playwright/test';

(async () => {
    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
    });
    const page = await browser.newPage();

    page.on('console', msg => console.log('PAGE:', msg.text()));
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    await page.goto('http://localhost:8000/pocs/simple_gpu_bench.html');

    await page.waitForFunction(() => {
        const status = document.getElementById('status').textContent;
        return status.includes('Complete') || status.includes('ERROR');
    }, { timeout: 60000 });

    const status = await page.$eval('#status', el => el.textContent);
    const resultsText = await page.$eval('#results', el => el.textContent);

    console.log('\n' + status);
    console.log(resultsText);

    await browser.close();
})();

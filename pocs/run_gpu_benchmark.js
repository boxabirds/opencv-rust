import { chromium } from '@playwright/test';

(async () => {
    console.log('GPU Performance Benchmark');
    console.log('='.repeat(80));
    console.log('Methodology: Measure pipeline overhead, run 100x, calculate net transform time');
    console.log('='.repeat(80));
    console.log('');

    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
    });
    const page = await browser.newPage();

    // Capture console messages
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('ERROR') || text.includes('Warning')) {
            console.log('PAGE:', text);
        }
    });

    // Capture errors
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

    await page.goto('http://localhost:8000/pocs/gpu_performance_benchmark.html');

    // Wait for benchmark to complete
    await page.waitForFunction(() => {
        const status = document.getElementById('status').textContent;
        return status.includes('complete') || status.includes('ERROR');
    }, { timeout: 120000 }); // 2 minutes timeout for large benchmarks

    // Get results
    const status = await page.$eval('#status', el => el.textContent);
    const resultsText = await page.$eval('#results', el => el.textContent);

    console.log('\n' + status);
    console.log('\n' + resultsText);
    console.log('\n' + '='.repeat(80));

    await browser.close();
})();

import { chromium } from '@playwright/test';

(async () => {
    console.log('GPU Batch Performance Benchmark');
    console.log('='.repeat(80));
    console.log('Warmup: 5 iterations | Benchmark: 100 iterations per operation');
    console.log('='.repeat(80));
    console.log('');

    const browser = await chromium.launch({
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
        headless: true
    });
    const context = await browser.newContext();
    context.setDefaultTimeout(300000); // 5 minute timeout
    const page = await context.newPage();

    // Show all console output for debugging
    page.on('console', msg => {
        const text = msg.text();
        console.log('PAGE:', text);
    });

    page.on('pageerror', err => console.error('❌ PAGE ERROR:', err.message));

    await page.goto('http://localhost:8000/pocs/gpu_batch_benchmark.html');

    // Wait for benchmark to complete (generous timeout for large images)
    await page.waitForFunction(() => {
        const status = document.getElementById('status').textContent;
        return status.includes('complete') || status.includes('ERROR');
    }, { timeout: 300000 }); // 5 minutes - options as second param

    const status = await page.$eval('#status', el => el.textContent);
    console.log('\n' + status);

    // Extract and format the table data
    const tableData = await page.evaluate(() => {
        const rows = Array.from(document.querySelectorAll('table tr'));
        return rows.map(row => {
            const cells = Array.from(row.querySelectorAll('td, th'));
            return cells.map(cell => ({
                text: cell.textContent.trim(),
                isHeader: cell.tagName === 'TH',
                isSizeHeader: cell.classList.contains('size-header')
            }));
        }).filter(row => row.length > 0);
    });

    // Format table for terminal
    console.log('\n');
    for (const row of tableData) {
        if (row[0].isSizeHeader) {
            console.log('\n' + '='.repeat(80));
            console.log('  ' + row[0].text);
            console.log('='.repeat(80));
        } else if (row[0].isHeader) {
            const line = row.map((cell, i) => {
                if (i === 0) return cell.text.padEnd(25);
                return cell.text.padStart(15);
            }).join(' ');
            console.log(line);
            console.log('-'.repeat(80));
        } else {
            const line = row.map((cell, i) => {
                if (i === 0) return cell.text.padEnd(25);
                return cell.text.padStart(15);
            }).join(' ');
            console.log(line);
        }
    }
    console.log('\n' + '='.repeat(80));

    await browser.close();
})();

import { chromium } from '@playwright/test';

(async () => {
  console.log('GPU Operations Test Suite\n');
  console.log('='.repeat(80));

  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('ERROR') || text.includes('Warning')) {
      console.log('PAGE:', text);
    }
  });

  await page.goto('http://localhost:8000/pocs/operation_test.html');

  // Wait for tests to complete
  await page.waitForFunction(() => {
    const status = document.getElementById('status').textContent;
    return status.includes('complete') || status.includes('Error');
  }, { timeout: 30000 });

  // Extract results
  const results = await page.evaluate(() => {
    const ops = Array.from(document.querySelectorAll('.operation'));
    return ops.map(op => {
      const name = op.querySelector('h3')?.textContent || 'Unknown';
      const error = op.querySelector('.fail');
      if (error) {
        return { name, error: error.textContent };
      }

      const rows = Array.from(op.querySelectorAll('tr'));
      const data = {};
      rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length === 2) {
          const key = cells[0].textContent.trim();
          const value = cells[1].textContent.trim();
          data[key] = value;
        }
      });

      return { name, ...data };
    });
  });

  // Print results table
  console.log('\nOperation                          | Accuracy | Max Diff | Speedup');
  console.log('-'.repeat(80));

  for (const result of results) {
    if (result.error) {
      console.log(`${result.name.padEnd(34)} | FAILED: ${result.error}`);
    } else {
      const name = result.name.padEnd(34);
      const accuracy = (result.Accuracy || 'N/A').padEnd(8);
      const maxDiff = (result['Max difference'] || 'N/A').padEnd(8);
      const speedup = result.Speedup || 'N/A';
      console.log(`${name} | ${accuracy} | ${maxDiff} | ${speedup}`);
    }
  }

  console.log('='.repeat(80));

  await browser.close();
})();

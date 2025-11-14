/**
 * Check which demo IDs from demoRegistry don't have cases in runDemo
 */

import { demos } from './src/demos/demoRegistry.js';
import fs from 'fs';

// Read App.jsx and extract all case statements
const appJsx = fs.readFileSync('./src/App.jsx', 'utf-8');

// Extract all case statements from runDemo function
const caseMatches = appJsx.matchAll(/case '([a-z_0-9]+)':/g);
const implementedCases = new Set();
for (const match of caseMatches) {
  implementedCases.add(match[1]);
}

console.log('='.repeat(80));
console.log('CHECKING FOR MISSING IMPLEMENTATIONS');
console.log('='.repeat(80));
console.log(`\nTotal demos in registry: ${demos.length}`);
console.log(`Total case statements in runDemo: ${implementedCases.size}\n`);

// Find missing implementations
const missingImpls = [];
const hasImpl = [];

for (const demo of demos) {
  if (implementedCases.has(demo.id)) {
    hasImpl.push(demo);
  } else {
    missingImpls.push(demo);
  }
}

if (missingImpls.length > 0) {
  console.log('='.repeat(80));
  console.log(`MISSING IMPLEMENTATIONS (${missingImpls.length})`);
  console.log('='.repeat(80));

  // Group by category
  const byCategory = {};
  for (const demo of missingImpls) {
    if (!byCategory[demo.category]) {
      byCategory[demo.category] = [];
    }
    byCategory[demo.category].push(demo);
  }

  for (const [category, items] of Object.entries(byCategory)) {
    console.log(`\n${category.toUpperCase()} (${items.length}):`);
    for (const item of items) {
      console.log(`  - ${item.id} (${item.name})`);
    }
  }
} else {
  console.log('✓ All demos have implementations!');
}

console.log('\n' + '='.repeat(80));
console.log(`SUMMARY`);
console.log('='.repeat(80));
console.log(`Implemented: ${hasImpl.length}`);
console.log(`Missing: ${missingImpls.length}`);
console.log(`Coverage: ${((hasImpl.length / demos.length) * 100).toFixed(1)}%`);

// Save results
const results = {
  total: demos.length,
  implemented: hasImpl.length,
  missing: missingImpls.length,
  coverage: ((hasImpl.length / demos.length) * 100).toFixed(1),
  missingList: missingImpls.map(d => ({ id: d.id, name: d.name, category: d.category }))
};

fs.writeFileSync('missing-implementations.json', JSON.stringify(results, null, 2));
console.log('\n✓ Results saved to missing-implementations.json');

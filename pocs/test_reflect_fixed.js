// Test FIXED REFLECT_101 implementation
function fixedReflect101(x, width) {
    let sx = x;
    if (sx < 0) {
        sx = -sx;  // FIXED: removed -1
    } else if (sx >= width) {
        sx = 2 * width - sx - 2;  // FIXED: changed -1 to -2
    }
    return Math.max(0, Math.min(sx, width - 1));
}

console.log('\nFIXED REFLECT_101 implementation:');
console.log('Width = 5 (pixels 0,1,2,3,4)\n');

for (let x = -5; x <= 9; x++) {
    const reflected = fixedReflect101(x, 5);
    console.log(`x=${x.toString().padStart(2)} -> ${reflected}`);
}

console.log('\n\nChecking key values:');
const checks = [
    [-1, 1],
    [-2, 2],
    [5, 3],
    [6, 2]
];

let allCorrect = true;
for (const [x, expected] of checks) {
    const got = fixedReflect101(x, 5);
    const status = got === expected ? '✓' : '✗';
    console.log(`x=${x} -> ${got} (expected ${expected}) ${status}`);
    if (got !== expected) allCorrect = false;
}

console.log(allCorrect ? '\n✓ ALL CORRECT!' : '\n✗ STILL WRONG!');

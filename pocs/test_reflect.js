// Test REFLECT_101 implementation
function ourReflect101(x, width) {
    let sx = x;
    if (sx < 0) {
        sx = -sx - 1;
    } else if (sx >= width) {
        sx = 2 * width - sx - 1;
    }
    return Math.max(0, Math.min(sx, width - 1));
}

console.log('\nOur REFLECT_101 implementation:');
console.log('Width = 5 (pixels 0,1,2,3,4)\n');

for (let x = -5; x <= 9; x++) {
    const reflected = ourReflect101(x, 5);
    console.log(`x=${x.toString().padStart(2)} -> ${reflected}`);
}

console.log('\n\nExpected REFLECT_101 pattern:');
console.log('...4,3,2,1|0,1,2,3,4|3,2,1,0...');
console.log('Edge pixel NOT duplicated');
console.log('\nExpected:');
console.log('x=-1 -> 1');
console.log('x=-2 -> 2');
console.log('x=-3 -> 3');
console.log('x=-4 -> 4');
console.log('x=5 -> 3');
console.log('x=6 -> 2');
console.log('x=7 -> 1');
console.log('x=8 -> 0');

#!/usr/bin/env python3
"""
Fix GPU shaders to use byte-level access helpers

This script updates all GPU shaders to correctly handle byte-level
access to u32 storage buffers, fixing the "everything appears red" bug.

Root cause: RGBA bytes [R,G,B,A] are packed into u32 words, but shaders
were treating packed values as individual bytes.

Fix: Add read_byte() and write_byte() helpers to unpack bytes correctly.
"""

import os
import re
from pathlib import Path

# Read the byte access helpers
HELPERS_PATH = Path("src/gpu/shaders/byte_access_helpers.wgsl")
BYTE_ACCESS_HELPERS = HELPERS_PATH.read_text()

def fix_shader(shader_path: Path) -> bool:
    """
    Fix a single shader file to use byte access helpers

    Returns True if file was modified, False otherwise
    """
    print(f"Processing {shader_path.name}...")

    content = shader_path.read_text()

    # Check if already fixed
    if "read_byte" in content:
        print(f"  ⚠ Already fixed, skipping")
        return False

    # Check if this shader uses array<u32> buffers
    if "array<u32>" not in content:
        print(f"  ℹ No u32 arrays, skipping")
        return False

    # Insert helpers after the struct definitions but before functions
    # Find the first @compute or fn declaration
    insert_marker = re.search(r"(@compute|^fn\s)", content, re.MULTILINE)

    if not insert_marker:
        print(f"  ⚠ No insertion point found, skipping")
        return False

    insert_pos = insert_marker.start()

    # Insert helpers with separator comments
    new_content = (
        content[:insert_pos] +
        "\n// === Byte Access Helpers ===\n" +
        "// Required for correct RGBA byte extraction from u32 storage buffers\n\n" +
        BYTE_ACCESS_HELPERS +
        "\n// === End Byte Access Helpers ===\n\n" +
        content[insert_pos:]
    )

    # Now replace buffer access patterns
    # This is a simple pattern - real shaders may need manual review

    # Pattern 1: input[idx] -> read_byte(&input, idx)
    # Only replace if it's clearly accessing pixel data
    new_content = re.sub(
        r'\binput\[([^\]]+)\](?!\s*=)',  # input[x] but not input[x] =
        r'read_byte(&input, \1)',
        new_content
    )

    # Pattern 2: output[idx] = value -> write_byte(&output, idx, value)
    new_content = re.sub(
        r'\boutput\[([^\]]+)\]\s*=\s*([^;]+);',
        r'write_byte(&output, \1, \2);',
        new_content
    )

    # Write back if changed
    if new_content != content:
        shader_path.write_text(new_content)
        print(f"  ✓ Fixed")
        return True
    else:
        print(f"  ℹ No changes needed")
        return False

def main():
    """Fix all shaders in src/gpu/shaders/"""
    shaders_dir = Path("src/gpu/shaders")

    if not shaders_dir.exists():
        print(f"Error: {shaders_dir} not found")
        print("Run this script from the project root directory")
        return 1

    print("=" * 80)
    print("GPU Shader Byte Access Fix")
    print("=" * 80)
    print()

    # Find all .wgsl files except the helpers
    shader_files = sorted([
        f for f in shaders_dir.glob("*.wgsl")
        if f.name != "byte_access_helpers.wgsl"
    ])

    print(f"Found {len(shader_files)} shader files\n")

    fixed_count = 0

    for shader_path in shader_files:
        if fix_shader(shader_path):
            fixed_count += 1

    print()
    print("=" * 80)
    print(f"Summary: Fixed {fixed_count}/{len(shader_files)} shaders")
    print("=" * 80)
    print()

    if fixed_count > 0:
        print("⚠ IMPORTANT: Manual review recommended!")
        print()
        print("This script applies basic patterns. Some shaders may need manual fixes:")
        print("  - Complex buffer indexing expressions")
        print("  - Buffers with different naming (src, dst, data, etc.)")
        print("  - Special cases (integral images, histograms, etc.)")
        print()
        print("Next steps:")
        print("  1. Review changes: git diff src/gpu/shaders/")
        print("  2. Build and test: wasm-pack build --features gpu,wasm")
        print("  3. Test in browser - colors should appear correctly now")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/bin/bash
# Check for type mismatches between GPU operations and WASM wrappers

echo "==================================================================="
echo "CHECKING FOR TYPE MISMATCHES IN GPU/WASM OPERATIONS"
echo "==================================================================="
echo ""

# Check all async GPU functions
for gpu_file in ../../src/gpu/ops/*.rs; do
    if [ -f "$gpu_file" ]; then
        basename=$(basename "$gpu_file" .rs)

        # Extract output Mat depth from GPU implementation
        gpu_depth=$(grep -A 20 "pub async fn.*_gpu_async" "$gpu_file" | grep -oE "MatDepth::(U8|F32|I32)" | head -1)

        if [ -n "$gpu_depth" ]; then
            echo "GPU operation: $basename"
            echo "  Output type: $gpu_depth"

            # Try to find corresponding WASM wrapper
            wasm_matches=$(grep -r "::${basename}_gpu_async" ../../src/wasm/ 2>/dev/null)

            if [ -n "$wasm_matches" ]; then
                # Extract the WASM file
                wasm_file=$(echo "$wasm_matches" | cut -d: -f1 | head -1)

                if [ -f "$wasm_file" ]; then
                    # Look for Mat::new near the GPU call
                    context=$(grep -B 10 -A 2 "::${basename}_gpu_async" "$wasm_file" | grep -E "Mat::new.*MatDepth::")

                    if [ -n "$context" ]; then
                        wasm_depth=$(echo "$context" | grep -oE "MatDepth::(U8|F32|I32)" | head -1)
                        echo "  WASM wrapper: $(basename $wasm_file)"
                        echo "  WASM type: $wasm_depth"

                        if [ "$gpu_depth" != "$wasm_depth" ]; then
                            echo "  ⚠️  TYPE MISMATCH DETECTED!"
                        else
                            echo "  ✓ Types match"
                        fi
                    fi
                fi
            fi
            echo ""
        fi
    fi
done

echo "==================================================================="
echo "SUMMARY: Check warnings above for type mismatches"
echo "==================================================================="

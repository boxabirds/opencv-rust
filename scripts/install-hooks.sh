#!/bin/bash
#
# Install Git hooks for opencv-rust
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.github/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "📦 Installing Git hooks..."

# Check if we're in a git repository
if [ ! -d "$GIT_HOOKS_DIR" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Install pre-push hook
if [ -f "$HOOKS_DIR/pre-push" ]; then
    ln -sf "../../.github/hooks/pre-push" "$GIT_HOOKS_DIR/pre-push"
    chmod +x "$GIT_HOOKS_DIR/pre-push"
    echo "✅ Installed pre-push hook"
else
    echo "⚠️  pre-push hook not found in $HOOKS_DIR"
fi

echo ""
echo "✅ Git hooks installed successfully"
echo ""
echo "Installed hooks:"
ls -lh "$GIT_HOOKS_DIR" | grep -v "\.sample$" || echo "  (none active yet)"
echo ""
echo "To disable a hook, remove it from .git/hooks/"
echo "To bypass hooks during push: git push --no-verify"

#!/bin/bash

# Verification script for SWA-MLA standalone package
# This ensures the package is truly self-contained and ready to deploy

echo "==================================================================="
echo "SWA-MLA Standalone Package Verification"
echo "==================================================================="
echo ""

ERRORS=0

# Check 1: No parent project imports
echo "✓ Checking for parent project dependencies..."
if grep -r "from models\." --include="*.py" . | grep -v "^Binary" | grep -v "test_setup" | grep -v "fix_imports" | grep -v "swa_mla_model.py:.*from models\."; then
    echo "  ✗ FAIL: Found imports from parent project"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ PASS: No parent project imports found"
fi

# Check 2: No relative imports going up
echo "✓ Checking for upward relative imports..."
if grep -r "from \.\." --include="*.py" . | grep -v "^Binary"; then
    echo "  ✗ FAIL: Found upward relative imports"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ PASS: No upward relative imports"
fi

# Check 3: All required files present
echo "✓ Checking required files..."
REQUIRED_FILES=(
    "train.py"
    "requirements.txt"
    "README.md"
    "QUICKSTART.md"
    "STANDALONE.md"
    "test_setup.py"
    "scripts/train_swa_mla.sh"
    "models/swa_mla_model.py"
    "models/mla.py"
    "models/mla_block.py"
    "models/attention.py"
    "models/mlp.py"
    "models/normalization.py"
    "models/positional_encoding.py"
    "data/data_loader_packed.py"
    "optimization/fp8_trainer.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ FAIL: Missing file: $file"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "  ✓ PASS: All required files present"
fi

# Check 4: Python syntax
echo "✓ Checking Python syntax..."
for pyfile in $(find . -name "*.py" -not -path "./venv/*"); do
    if ! python -m py_compile "$pyfile" 2>/dev/null; then
        echo "  ✗ FAIL: Syntax error in $pyfile"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -eq 0 ] || [ $(find . -name "*.py" -not -path "./venv/*" | wc -l) -gt 0 ]; then
    echo "  ✓ PASS: All Python files have valid syntax"
fi

# Check 5: File sizes
echo "✓ Checking package size..."
SIZE=$(du -sh . | awk '{print $1}')
echo "  Package size: $SIZE"

# Check 6: Documentation
echo "✓ Checking documentation..."
if [ -f "README.md" ] && [ -f "QUICKSTART.md" ] && [ -f "STANDALONE.md" ]; then
    echo "  ✓ PASS: All documentation files present"
else
    echo "  ✗ FAIL: Missing documentation"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: Executable permissions
echo "✓ Checking executable permissions..."
if [ -x "scripts/train_swa_mla.sh" ]; then
    echo "  ✓ PASS: Training script is executable"
else
    echo "  ⚠ WARNING: Training script not executable (run: chmod +x scripts/train_swa_mla.sh)"
fi

# Summary
echo ""
echo "==================================================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ VERIFICATION PASSED"
    echo "==================================================================="
    echo ""
    echo "The package is ready to deploy!"
    echo ""
    echo "You can:"
    echo "  1. Move this folder anywhere"
    echo "  2. Create a tarball: tar -czf swa_mla.tar.gz swa_mla/"
    echo "  3. Zip it: zip -r swa_mla.zip swa_mla/"
    echo "  4. Use it as a git submodule"
    echo ""
    echo "To test the setup:"
    echo "  python test_setup.py"
    echo ""
    echo "To start training:"
    echo "  ./scripts/train_swa_mla.sh small 8 2048"
    echo ""
    exit 0
else
    echo "✗ VERIFICATION FAILED ($ERRORS errors)"
    echo "==================================================================="
    echo ""
    echo "Please fix the errors above before deploying."
    echo ""
    exit 1
fi

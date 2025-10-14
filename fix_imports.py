"""Fix imports in copied modules to make them standalone."""

import os
import re

def fix_imports_in_file(filepath, replacements):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed imports in {filepath}")
    else:
        print(f"No changes needed in {filepath}")

# Define import replacements
replacements = {
    'from models.blocks.normalization import': 'from normalization import',
    'from models.blocks.attention import': 'from attention import',
    'from models.blocks.mlp import': 'from mlp import',
    'from models.blocks.mla_block import': 'from mla_block import',
    'from models.blocks.mla import': 'from mla import',
    'from models.blocks.positional_encoding import': 'from positional_encoding import',
    'from models.models.mla_model import': 'from positional_encoding import',
    'from models.optimizers import': '# from optimizers import',
    'from .normalization import': 'from normalization import',
    'from .attention import': 'from attention import',
    'from .mlp import': 'from mlp import',
    'from .mla_block import': 'from mla_block import',
    'from .mla import': 'from mla import',
    'from .positional_encoding import': 'from positional_encoding import',
    'from .mla_fp8 import': '# from mla_fp8 import',
    'from .tensor_utils import': '# from tensor_utils import',
    'from models.blocks.mla_selective_fast import': '# from mla_selective_fast import',
}

# Fix imports in all model files
models_dir = 'models'
for filename in os.listdir(models_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        filepath = os.path.join(models_dir, filename)
        fix_imports_in_file(filepath, replacements)

print("\nImport fixing complete!")

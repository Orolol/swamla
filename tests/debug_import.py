
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

try:
    import moe_triton
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error during import: {e}")

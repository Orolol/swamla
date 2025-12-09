
try:
    import fast_loader
    print("fast_loader is installed")
    print(f"fast_loader version: {getattr(fast_loader, '__version__', 'unknown')}")
    from fast_loader import FastFinewebDataset
    print("FastFinewebDataset imported successfully")
except ImportError as e:
    print(f"Error importing fast_loader: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

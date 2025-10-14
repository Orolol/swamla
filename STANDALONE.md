# SWA-MLA Standalone Package

This is a **completely self-contained** implementation of the SWA-MLA hybrid model.

## What's Included

✓ **Complete Model Architecture**
- Sliding Window Attention (SWA) blocks
- Multi-head Latent Attention (MLA) blocks
- RoPE positional embeddings
- All normalization layers (RMSNorm, DynamicTanh)

✓ **Training Infrastructure**
- Packed sequence data loader (minimizes padding waste)
- FP8 mixed precision support (H100/H200)
- Multiple optimizer support (AdamW, Lion, FP8 variants)
- Multi-GPU DDP training
- Wandb integration
- Automatic checkpointing

✓ **Optimization Features**
- torch.compile compatibility
- Gradient checkpointing
- Mixed precision training
- Memory-efficient attention backends

✓ **Documentation**
- Complete README with examples
- Quick start guide
- Setup test script
- Training launch scripts

## Package Size

```
Total: ~372 KB (excluding dependencies)
- Models: ~200 KB
- Data loader: ~13 KB
- Training script: ~14 KB
- Optimization: ~15 KB
- Documentation: ~30 KB
```

## Zero External Dependencies

This package has **NO dependencies** on the parent project:
- ✓ All imports are local
- ✓ No `../` relative imports
- ✓ No shared utility modules
- ✓ Completely standalone `train.py`

## Export Ready

You can move this entire folder anywhere:

```bash
# Copy to a new location
cp -r swa_mla /path/to/new/location

# Or create a tarball
tar -czf swa_mla.tar.gz swa_mla/

# Or zip it
zip -r swa_mla.zip swa_mla/
```

Then in the new location:

```bash
cd swa_mla
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./scripts/train_swa_mla.sh small 8 2048
```

It just works!

## Differences from Main Project

| Feature | Main Project | Standalone |
|---------|-------------|------------|
| Dependencies | Shared modules | Self-contained |
| Model variants | 15+ architectures | SWA-MLA only |
| Data loaders | 7 types | Packed only |
| Optimizers | Full suite | Core + FP8 |
| Training modes | Lightning + Native | Native only |
| Size | ~50 MB | ~372 KB |
| Complexity | Research-grade | Production-ready |

## What's Not Included

To keep this standalone and focused, we **excluded**:
- ❌ Other model architectures (GPT, LLaDA, MoE, etc.)
- ❌ PyTorch Lightning wrapper
- ❌ Progressive training system
- ❌ Other data loaders (dynamic, concatenated, etc.)
- ❌ GaLore, APOLLO, Muon optimizers
- ❌ Model adaptation utilities
- ❌ Experimental features

**These exclusions make the code:**
- Easier to understand
- Faster to set up
- Simpler to maintain
- Ready for production

## Use Cases

This standalone package is perfect for:

1. **Production Deployment** - Small, focused, reliable
2. **Research Projects** - Quick to integrate and modify
3. **Learning** - Clear, well-documented implementation
4. **Benchmarking** - Consistent, reproducible setup
5. **Fine-tuning** - Start from trained checkpoint

## Verification

Run the test suite to verify everything works:

```bash
python test_setup.py
```

All tests should pass ✓

## Integration with Other Projects

### As a Git Submodule

```bash
git submodule add https://github.com/your-org/swa-mla.git
cd swa-mla
pip install -r requirements.txt
```

### As a Python Package

```python
import sys
sys.path.insert(0, 'path/to/swa_mla/models')

from swa_mla_model import create_swa_mla_model

model = create_swa_mla_model(size='small')
```

### As a Docker Container

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY swa_mla/ /app/
RUN pip install -r requirements.txt

CMD ["python", "train.py", "--size", "small"]
```

## Maintenance

This standalone package is:
- ✓ **Version locked** - Won't break with main project changes
- ✓ **Self-contained** - All dependencies explicit
- ✓ **Tested** - Includes verification scripts
- ✓ **Documented** - Multiple levels of documentation

## License

Same as parent project. This is a convenience extraction, not a fork.

## Support

- **Issues**: Test with `python test_setup.py` first
- **Questions**: See README.md and QUICKSTART.md
- **Bugs**: File issues with test_setup.py output
- **Features**: This is a focused extraction - keep it simple!

---

**This is exactly what you need, nothing more.**

Enjoy training! 🚀

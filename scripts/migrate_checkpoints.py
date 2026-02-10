#!/usr/bin/env python3
"""Migrate old-format checkpoints to the new base/instruct naming convention.

Local format:
  Old:      checkpoint_{tokens}_step{step}.pt
  Base:     checkpoint_base_{tokens}_step{step}.pt
  Instruct: checkpoint_instruct_pt{pretrain}_it{instruct}_step{step}.pt

HuggingFace format:
  Old:      checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin
  Old:      instruct/checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin
  Base:     base/checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin
  Instruct: instruct/checkpoint_pt{pretrain}_it{instruct}_loss_{loss}/pytorch_model.bin

Usage:
    # Local checkpoints
    python scripts/migrate_checkpoints.py outputs/engram-moe/
    python scripts/migrate_checkpoints.py outputs/engram-moe/ --instruct --pretrain-tokens 2B
    python scripts/migrate_checkpoints.py outputs/engram-moe/ --dry-run

    # HuggingFace checkpoints
    python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe
    python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe --dry-run
    python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe --instruct --pretrain-tokens 2B
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path


def format_tokens(n: int) -> str:
    """Format token count with appropriate suffix for display."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def parse_token_count(s: str) -> int:
    """Parse a token string like '2B', '500M', '100K' into an integer."""
    s = s.strip()
    multipliers = [
        ('B', 1_000_000_000), ('b', 1_000_000_000),
        ('M', 1_000_000), ('m', 1_000_000),
        ('K', 1_000), ('k', 1_000),
    ]
    for suffix, mult in multipliers:
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


# ---------------------------------------------------------------------------
# Local checkpoint migration
# ---------------------------------------------------------------------------

def find_old_checkpoints(directory: str) -> list[dict]:
    """Find checkpoint files using the old naming convention."""
    results = []
    # Match: checkpoint_{tokens}_step{step}.pt but NOT checkpoint_base_* or checkpoint_instruct_*
    old_pattern = re.compile(r'^checkpoint_(.+)_step(\d+)\.pt$')

    for fname in os.listdir(directory):
        if fname.startswith('checkpoint_base_') or fname.startswith('checkpoint_instruct_'):
            continue  # Already migrated
        match = old_pattern.match(fname)
        if match:
            results.append({
                'filename': fname,
                'filepath': os.path.join(directory, fname),
                'tokens_str': match.group(1),
                'step': int(match.group(2)),
            })

    results.sort(key=lambda x: x['step'])
    return results


def migrate_local_checkpoint(
    filepath: str,
    checkpoint_type: str = "base",
    pretrain_tokens_override: int | None = None,
    dry_run: bool = False,
) -> str | None:
    """Migrate a single local checkpoint file.

    Returns:
        New filepath if migrated, None if skipped
    """
    import torch

    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ERROR: Failed to load {filename}: {e}")
        return None

    # Skip if already has new metadata
    if 'checkpoint_type' in checkpoint:
        print(f"  SKIP: {filename} (already has checkpoint_type='{checkpoint['checkpoint_type']}')")
        return None

    total_tokens = checkpoint.get('total_tokens', 0)
    step = checkpoint.get('step', 0)

    # Determine token counts based on type
    # NOTE: In the old format, instruct training resets total_tokens to 0
    # at the start, so total_tokens in old instruct checkpoints = instruct tokens only.
    if checkpoint_type == "instruct":
        if pretrain_tokens_override is not None:
            pretrain_tokens = pretrain_tokens_override
        else:
            ckpt_args = checkpoint.get('args', {})
            pretrain_tokens = ckpt_args.get('pretrain_tokens', 0)
            if pretrain_tokens == 0:
                print(f"  WARNING: {filename} — cannot determine pretrain_tokens for instruct checkpoint.")
                print(f"           Use --pretrain-tokens to specify. Defaulting to 0.")
        instruct_tokens = total_tokens
        new_total = pretrain_tokens + instruct_tokens
    else:
        pretrain_tokens = total_tokens
        instruct_tokens = 0
        new_total = total_tokens

    # Build new filename
    if checkpoint_type == "instruct":
        pt_str = format_tokens(pretrain_tokens).replace('.', '_')
        it_str = format_tokens(instruct_tokens).replace('.', '_')
        new_filename = f"checkpoint_instruct_pt{pt_str}_it{it_str}_step{step}.pt"
    else:
        tokens_str = format_tokens(pretrain_tokens).replace('.', '_')
        new_filename = f"checkpoint_base_{tokens_str}_step{step}.pt"

    new_filepath = os.path.join(directory, new_filename)

    if dry_run:
        print(f"  WOULD: {filename} -> {new_filename}")
        print(f"         type={checkpoint_type}, pretrain={format_tokens(pretrain_tokens)}, instruct={format_tokens(instruct_tokens)}")
        return new_filepath

    # Add new metadata
    checkpoint['checkpoint_type'] = checkpoint_type
    checkpoint['pretrain_tokens'] = pretrain_tokens
    checkpoint['instruct_tokens'] = instruct_tokens
    checkpoint['total_tokens'] = new_total

    # Save updated checkpoint with new name
    torch.save(checkpoint, new_filepath)
    print(f"  SAVED: {new_filename}")
    print(f"         type={checkpoint_type}, pretrain={format_tokens(pretrain_tokens)}, instruct={format_tokens(instruct_tokens)}")

    # Remove old file if different name
    if new_filepath != filepath:
        os.remove(filepath)
        print(f"  REMOVED: {filename}")

    return new_filepath


def run_local_migration(args):
    """Migrate local checkpoint files."""
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    checkpoint_type = "instruct" if args.instruct else "base"
    pretrain_tokens_override = parse_token_count(args.pretrain_tokens) if args.pretrain_tokens else None

    if args.instruct and pretrain_tokens_override is None:
        print("WARNING: --instruct without --pretrain-tokens — pretrain_tokens will default to 0")
        print("         Use --pretrain-tokens to specify the base model's training tokens\n")

    old_checkpoints = find_old_checkpoints(args.directory)

    if not old_checkpoints:
        print(f"No old-format checkpoints found in {args.directory}")
        all_pt = [f for f in os.listdir(args.directory) if f.endswith('.pt')]
        if all_pt:
            base_count = sum(1 for f in all_pt if f.startswith('checkpoint_base_'))
            instruct_count = sum(1 for f in all_pt if f.startswith('checkpoint_instruct_'))
            print(f"Found: {base_count} base, {instruct_count} instruct checkpoints (already migrated)")
        sys.exit(0)

    print(f"Found {len(old_checkpoints)} old-format checkpoint(s) in {args.directory}")
    print(f"Migration type: {checkpoint_type}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")
    else:
        print()

    migrated = 0
    for ckpt in old_checkpoints:
        result = migrate_local_checkpoint(
            ckpt['filepath'],
            checkpoint_type=checkpoint_type,
            pretrain_tokens_override=pretrain_tokens_override,
            dry_run=args.dry_run,
        )
        if result:
            migrated += 1
        print()

    print(f"{'Would migrate' if args.dry_run else 'Migrated'}: {migrated}/{len(old_checkpoints)} checkpoints")


# ---------------------------------------------------------------------------
# HuggingFace checkpoint migration
# ---------------------------------------------------------------------------

def find_old_hf_checkpoints(files: list[str]) -> list[dict]:
    """Find old-format checkpoint directories in an HF repo.

    Old formats:
      - checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin  (root, base)
      - instruct/checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin  (old instruct)

    New formats (skipped):
      - base/checkpoint_tokens_{tokens}_loss_{loss}/pytorch_model.bin
      - instruct/checkpoint_pt{pt}_it{it}_loss_{loss}/pytorch_model.bin
    """
    results = []

    # Old root-level base pattern
    root_pattern = re.compile(
        r'^checkpoint_tokens_(?P<tokens>[\d.]+[kKmMbB]?)_loss_(?P<loss>[\d.]+)/pytorch_model\.bin$'
    )
    # Old instruct pattern
    old_instruct_pattern = re.compile(
        r'^instruct/checkpoint_tokens_(?P<tokens>[\d.]+[kKmMbB]?)_loss_(?P<loss>[\d.]+)/pytorch_model\.bin$'
    )

    for f in files:
        # Skip new-format paths
        if f.startswith('base/') or f.startswith('instruct/checkpoint_pt'):
            continue

        match = root_pattern.match(f)
        if match:
            results.append({
                'file': f,
                'dir': f.rsplit('/', 1)[0],
                'tokens_str': match.group('tokens'),
                'loss': match.group('loss'),
                'source': 'root',
            })
            continue

        match = old_instruct_pattern.match(f)
        if match:
            results.append({
                'file': f,
                'dir': f.rsplit('/', 1)[0],
                'tokens_str': match.group('tokens'),
                'loss': match.group('loss'),
                'source': 'instruct',
            })

    return results


def find_new_hf_checkpoints(files: list[str]) -> list[dict]:
    """Find already-migrated checkpoint directories in an HF repo."""
    results = []
    base_pattern = re.compile(
        r'^base/checkpoint_tokens_(?P<tokens>[\d.]+[kKmMbB]?)_loss_(?P<loss>[\d.]+)/pytorch_model\.bin$'
    )
    instruct_pattern = re.compile(
        r'^instruct/checkpoint_pt(?P<pt>[\d.]+[kKmMbB]?)_it(?P<it>[\d.]+[kKmMbB]?)_loss_(?P<loss>[\d.]+)/pytorch_model\.bin$'
    )

    for f in files:
        if base_pattern.match(f):
            results.append({'file': f, 'type': 'base'})
        elif instruct_pattern.match(f):
            results.append({'file': f, 'type': 'instruct'})

    return results


def migrate_hf_checkpoint(
    api,
    repo_id: str,
    hf_token: str,
    ckpt: dict,
    checkpoint_type: str,
    pretrain_tokens_override: int | None,
    dry_run: bool,
    delete_old: bool,
) -> bool:
    """Migrate a single HF checkpoint directory.

    Downloads pytorch_model.bin, adds metadata, re-uploads under new path.

    Returns:
        True if migrated, False if skipped/errored
    """
    import torch
    from huggingface_hub import hf_hub_download

    old_dir = ckpt['dir']
    loss_str = ckpt['loss']

    # Determine the effective type for this checkpoint
    # Root checkpoints are base by default, instruct/ checkpoints are instruct
    if ckpt['source'] == 'instruct':
        eff_type = 'instruct'
    else:
        eff_type = checkpoint_type  # User-specified or default 'base'

    print(f"\n  [{old_dir}]")

    # Download the pytorch_model.bin
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=ckpt['file'], token=hf_token)
    except Exception as e:
        print(f"  ERROR: Failed to download {ckpt['file']}: {e}")
        return False

    # Load checkpoint
    try:
        checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ERROR: Failed to load {ckpt['file']}: {e}")
        return False

    # Skip if already has metadata
    if 'checkpoint_type' in checkpoint:
        print(f"  SKIP: already has checkpoint_type='{checkpoint['checkpoint_type']}'")
        return False

    total_tokens = checkpoint.get('total_tokens', 0)

    # Compute token counts
    if eff_type == "instruct":
        if pretrain_tokens_override is not None:
            pretrain_tokens = pretrain_tokens_override
        else:
            ckpt_args = checkpoint.get('args', {})
            pretrain_tokens = ckpt_args.get('pretrain_tokens', 0)
            if pretrain_tokens == 0:
                print(f"  WARNING: cannot determine pretrain_tokens. Use --pretrain-tokens. Defaulting to 0.")
        instruct_tokens = total_tokens
        new_total = pretrain_tokens + instruct_tokens
    else:
        pretrain_tokens = total_tokens
        instruct_tokens = 0
        new_total = total_tokens

    # Build new HF path
    if eff_type == "instruct":
        pt_str = format_tokens(pretrain_tokens)
        it_str = format_tokens(instruct_tokens)
        new_dir = f"instruct/checkpoint_pt{pt_str}_it{it_str}_loss_{loss_str}"
    else:
        tokens_str = format_tokens(total_tokens)
        new_dir = f"base/checkpoint_tokens_{tokens_str}_loss_{loss_str}"

    print(f"  {old_dir} -> {new_dir}")
    print(f"  type={eff_type}, pretrain={format_tokens(pretrain_tokens)}, instruct={format_tokens(instruct_tokens)}")

    if dry_run:
        print(f"  WOULD upload to {new_dir}/")
        if delete_old:
            print(f"  WOULD delete {old_dir}/")
        return True

    # Add metadata to checkpoint
    checkpoint['checkpoint_type'] = eff_type
    checkpoint['pretrain_tokens'] = pretrain_tokens
    checkpoint['instruct_tokens'] = instruct_tokens
    checkpoint['total_tokens'] = new_total

    # Save to temp file, upload, then clean up
    with tempfile.TemporaryDirectory() as tmpdir:
        # We need to re-upload ALL files from the old dir, not just pytorch_model.bin
        # List all files under the old directory
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(repo_id, token=hf_token)
        old_prefix = old_dir + "/"
        dir_files = [f for f in all_files if f.startswith(old_prefix)]

        # Download all files from old dir
        for rel_file in dir_files:
            rel_name = rel_file[len(old_prefix):]
            local_file = hf_hub_download(repo_id=repo_id, filename=rel_file, token=hf_token)

            dest = os.path.join(tmpdir, rel_name)
            os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else tmpdir, exist_ok=True)

            if rel_name == "pytorch_model.bin":
                # Save the modified checkpoint
                torch.save(checkpoint, dest)
            else:
                # Copy other files (tokenizer, config.json, README) as-is
                import shutil
                shutil.copy2(local_file, dest)

        # Also update config.json if it exists
        config_path = os.path.join(tmpdir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['checkpoint_type'] = eff_type
            config['pretrain_tokens'] = pretrain_tokens
            config['instruct_tokens'] = instruct_tokens
            config['total_tokens'] = new_total
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        # Upload to new path
        try:
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=new_dir,
                commit_message=f"Migrate checkpoint: {old_dir} -> {new_dir}",
            )
            print(f"  UPLOADED: {new_dir}/")
        except Exception as e:
            print(f"  ERROR: Failed to upload to {new_dir}: {e}")
            return False

    # Delete old directory
    if delete_old:
        try:
            api.delete_folder(
                path_in_repo=old_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Remove old checkpoint: {old_dir} (migrated to {new_dir})",
            )
            print(f"  DELETED: {old_dir}/")
        except Exception as e:
            print(f"  WARNING: Failed to delete old dir {old_dir}: {e}")
            print(f"           You may want to delete it manually.")

    return True


def run_hf_migration(args):
    """Migrate HuggingFace repo checkpoints."""
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError:
        print("Error: huggingface_hub is required for HF migration.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    repo_id = args.hf_repo
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        print("Set it with: export HF_TOKEN=hf_...")
        sys.exit(1)

    checkpoint_type = "instruct" if args.instruct else "base"
    pretrain_tokens_override = parse_token_count(args.pretrain_tokens) if args.pretrain_tokens else None

    if args.instruct and pretrain_tokens_override is None:
        print("WARNING: --instruct without --pretrain-tokens — pretrain_tokens will default to 0")
        print("         Use --pretrain-tokens to specify the base model's training tokens\n")

    print(f"Scanning HF repo: {repo_id}")
    try:
        files = list_repo_files(repo_id, token=hf_token)
    except Exception as e:
        print(f"Error: Failed to list repo files: {e}")
        sys.exit(1)

    # Find old-format checkpoints
    old_checkpoints = find_old_hf_checkpoints(files)

    # Show existing new-format checkpoints
    new_checkpoints = find_new_hf_checkpoints(files)
    if new_checkpoints:
        base_count = sum(1 for c in new_checkpoints if c['type'] == 'base')
        instruct_count = sum(1 for c in new_checkpoints if c['type'] == 'instruct')
        print(f"Already migrated: {base_count} base, {instruct_count} instruct")

    if not old_checkpoints:
        print(f"No old-format checkpoints found in {repo_id}")
        sys.exit(0)

    # Group by source
    root_ckpts = [c for c in old_checkpoints if c['source'] == 'root']
    instruct_ckpts = [c for c in old_checkpoints if c['source'] == 'instruct']

    print(f"\nFound {len(old_checkpoints)} old-format checkpoint(s):")
    if root_ckpts:
        print(f"  Root (-> {checkpoint_type}): {len(root_ckpts)}")
        for c in root_ckpts:
            print(f"    {c['dir']}")
    if instruct_ckpts:
        print(f"  instruct/ (-> instruct): {len(instruct_ckpts)}")
        for c in instruct_ckpts:
            print(f"    {c['dir']}")

    if args.dry_run:
        print("\nDRY RUN — no files will be modified\n")
    else:
        print()

    api = HfApi(token=hf_token)
    migrated = 0

    for ckpt in old_checkpoints:
        result = migrate_hf_checkpoint(
            api=api,
            repo_id=repo_id,
            hf_token=hf_token,
            ckpt=ckpt,
            checkpoint_type=checkpoint_type,
            pretrain_tokens_override=pretrain_tokens_override,
            dry_run=args.dry_run,
            delete_old=args.delete_old,
        )
        if result:
            migrated += 1

    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'}: {migrated}/{len(old_checkpoints)} checkpoints")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Migrate checkpoints to new base/instruct naming convention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local checkpoints
  python scripts/migrate_checkpoints.py outputs/engram-moe/
  python scripts/migrate_checkpoints.py outputs/engram-moe/ --instruct --pretrain-tokens 2B
  python scripts/migrate_checkpoints.py outputs/engram-moe/ --dry-run

  # HuggingFace checkpoints
  python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe --dry-run
  python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe --delete-old
  python scripts/migrate_checkpoints.py --hf-repo orolol/swamla-engram-moe --instruct --pretrain-tokens 2B
""",
    )
    parser.add_argument('directory', type=str, nargs='?', default=None,
                        help='Directory containing local checkpoint files')
    parser.add_argument('--hf-repo', type=str, default=None,
                        help='HuggingFace repo ID to migrate (e.g., orolol/swamla-engram-moe)')
    parser.add_argument('--instruct', action='store_true',
                        help='Treat root checkpoints as instruct (default: base)')
    parser.add_argument('--pretrain-tokens', type=str, default=None,
                        help='Pretrain tokens for instruct checkpoints (e.g., "2B", "500M")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--delete-old', action='store_true',
                        help='(HF only) Delete old checkpoint directories after migration')

    args = parser.parse_args()

    if args.hf_repo:
        run_hf_migration(args)
    elif args.directory:
        run_local_migration(args)
    else:
        parser.error("Either a local directory or --hf-repo must be specified")


if __name__ == '__main__':
    main()

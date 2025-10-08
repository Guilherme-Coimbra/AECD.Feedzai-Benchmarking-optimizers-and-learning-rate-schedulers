import json
from pathlib import Path

NB_PATH = Path('notebooks/03_baf_baseline.ipynb')


def join(c):
    return ''.join(c.get('source', []))


def split_lines(s):
    # Keep trailing newlines in each element as in ipynb format
    return [line if line.endswith('\n') else line + '\n' for line in s.splitlines()]


def ensure_comment_block(lines, anchor, insert_after=True, block_lines=None, once_marker=None):
    """
    Insert block_lines around anchor match (first occurrence). If once_marker is present
    anywhere in lines, skip to avoid duplicate insertion.
    """
    if not block_lines:
        return lines
    text = ''.join(lines)
    if once_marker and once_marker in text:
        return lines

    try:
        idx = next(i for i, l in enumerate(lines) if anchor in l)
    except StopIteration:
        return lines

    insert_idx = idx + 1 if insert_after else idx
    return lines[:insert_idx] + block_lines + lines[insert_idx:]


def annotate():
    nb = json.loads(NB_PATH.read_text())

    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        src = join(cell)

        # 1) EmbeddingStack: explain trainable weights and updates
        if 'class EmbeddingStack' in src and 'self.embs = nn.ModuleList' in src:
            lines = split_lines(src)
            indent = '        '
            block = [
                indent + '# Each nn.Embedding holds a trainable weight matrix [num_categories, emb_dim].\n',
                indent + '# Gradients flow through the lookup; AdamW (below) updates these per batch.\n',
                indent + '# No padding_idx here; all IDs are assumed valid from label encoding.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='self.output_dims =',
                insert_after=True,
                block_lines=block,
                once_marker='Gradients flow through the lookup'
            )
            cell['source'] = lines
            continue

        # 2) Instantiation of emb_stack: clarify optimization target
        if 'emb_stack = EmbeddingStack(cardinality, emb_dims)' in src:
            lines = split_lines(src)
            block = [
                '# Instantiate embeddings; their weight matrices will receive gradients\n',
                '# from the loss and be updated by AdamW during training.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='emb_stack = EmbeddingStack',
                insert_after=False,
                block_lines=block,
                once_marker='their weight matrices will receive gradients'
            )
            cell['source'] = lines
            continue

        # 3) Config cell: explain STEPS_PER_EPOCH/NUM_EPOCHS -> schedulers total steps
        if 'BATCH_SIZE' in src and 'POS_FRAC' in src and 'STEPS_PER_EPOCH' in src and 'NUM_EPOCHS' in src:
            lines = split_lines(src)
            block = [
                '# Note: LR schedulers step per-batch; total steps = STEPS_PER_EPOCH * NUM_EPOCHS.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='STEPS_PER_EPOCH',
                insert_after=False,
                block_lines=block,
                once_marker='LR schedulers step per-batch'
            )
            cell['source'] = lines
            continue

        # 4) FFN build: explain residuals, device placement
        if 'ffn = FeedforwardNetwork(' in src and 'emb_stack = emb_stack.to(device)' in src:
            lines = split_lines(src)
            block_residual = [
                '# Residual spec: takes tensor just before layer 1 and adds it to the\n',
                '# output after layer 2 (pre1 -> post2), improving gradient flow.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='residuals = [',
                insert_after=False,
                block_lines=block_residual,
                once_marker='pre1 -> post2), improving gradient flow'
            )
            block_device = [
                '# Ensure both modules live on the same device as the data.\n',
                '# Their parameters receive gradients and updates on this device.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='emb_stack = emb_stack.to(device)',
                insert_after=False,
                block_lines=block_device,
                once_marker='modules live on the same device'
            )
            cell['source'] = lines
            continue

        # 5) Loss: explain pos_weight impact
        if 'POS_WEIGHT' in src and 'weighted_bce_with_logits' in src:
            lines = split_lines(src)
            block = [
                '# pos_weight scales positive examples inside BCEWithLogits;\n',
                '# gradients for positive samples are amplified by POS_WEIGHT.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='POS_WEIGHT',
                insert_after=True,
                block_lines=block,
                once_marker='gradients for positive samples are amplified'
            )
            cell['source'] = lines
            continue

        # 6) Optimizers & schedulers: document groups and when updates happen
        if 'opt_adamw = torch.optim.AdamW' in src and 'opt_sgd   = torch.optim.SGD' in src:
            # Direct string replacements for the two small helper fns to guarantee insertion
            # Line-based injection for robust handling of whitespace
            lines_raw = src.splitlines(True)
            # zero_grads comment
            if any('def zero_grads' in l for l in lines_raw) and not any('Reset accumulated gradients' in l for l in lines_raw):
                for j,l in enumerate(lines_raw):
                    if 'def zero_grads' in l:
                        lines_raw.insert(j+1, '    # Reset accumulated gradients for both optimizers/groups\n')
                        break
            # step_optimizers_and_schedulers comments
            if any('def step_optimizers_and_schedulers' in l for l in lines_raw) and not any('Apply parameter updates' in l for l in lines_raw):
                for j,l in enumerate(lines_raw):
                    if 'def step_optimizers_and_schedulers' in l:
                        lines_raw.insert(j+1, '    # Apply parameter updates\n')
                        lines_raw.insert(j+2, '    # - AdamW updates embedding matrices and output head weights\n')
                        lines_raw.insert(j+3, '    # - SGD updates hidden MLP weights\n')
                        break
                # LR scheduling comment before first sched step
                for j,l in enumerate(lines_raw):
                    if 'sched_adamw.step' in l:
                        lines_raw.insert(j, '    # Per-batch LR scheduling for both optimizers\n')
                        break
            src = ''.join(lines_raw)
            lines = split_lines(src)
            # Param group annotations
            lines = ensure_comment_block(
                lines,
                anchor='emb_params =',
                insert_after=False,
                block_lines=['# Embedding matrices (one per categorical feature)\n'],
                once_marker='Embedding matrices (one per categorical feature)'
            )
            lines = ensure_comment_block(
                lines,
                anchor='out_params =',
                insert_after=False,
                block_lines=['# Output/classification head parameters\n'],
                once_marker='Output/classification head parameters'
            )
            lines = ensure_comment_block(
                lines,
                anchor='hidden_params =',
                insert_after=False,
                block_lines=['# Hidden MLP layers parameters (all but output)\n'],
                once_marker='Hidden MLP layers parameters'
            )
            # Explain which optimizer updates what
            lines = ensure_comment_block(
                lines,
                anchor='adamw_params =',
                insert_after=False,
                block_lines=['# Optimized by AdamW: embeddings + output layer\n'],
                once_marker='Optimized by AdamW: embeddings + output layer'
            )
            lines = ensure_comment_block(
                lines,
                anchor='sgd_params   =',
                insert_after=False,
                block_lines=['# Optimized by SGD: hidden layers only\n'],
                once_marker='Optimized by SGD: hidden layers only'
            )
            # zero_grads explanation
            lines = ensure_comment_block(
                lines,
                anchor='def zero_grads',
                insert_after=True,
                block_lines=['    # Reset accumulated gradients for both optimizers/groups\n'],
                once_marker='Reset accumulated gradients for both optimizers'
            )
            # step explanation
            step_block = [
                '    # Apply parameter updates\n',
                '    # - AdamW updates embedding matrices and output head weights\n',
                '    # - SGD updates hidden MLP weights\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='def step_optimizers_and_schedulers',
                insert_after=True,
                block_lines=step_block,
                once_marker='AdamW updates embedding matrices and output head weights'
            )
            lines = ensure_comment_block(
                lines,
                anchor='sched_adamw.step',
                insert_after=False,
                block_lines=['    # Per-batch LR scheduling for both optimizers\n'],
                once_marker='Per-batch LR scheduling for both optimizers'
            )
            cell['source'] = lines
            continue

        # 7) StratifiedBatchSampler: explain sampling with replacement
        if 'class StratifiedBatchSampler' in src:
            lines = split_lines(src)
            block = [
                '        # Build class-balanced batches via sampling with replacement\n',
                '        # Ensures ~pos_per_batch positives per batch across num_batches.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='def __iter__',
                insert_after=True,
                block_lines=block,
                once_marker='Build class-balanced batches via sampling with replacement'
            )
            cell['source'] = lines
            continue

        # 8) Dataset: clarify label dtype and usage
        if 'class CatsNumsDataset' in src:
            lines = split_lines(src)
            block = [
                '        # Labels are stored as long here but cast to float in the\n',
                '        # training loop for BCEWithLogits loss.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='self.y      =',
                insert_after=True,
                block_lines=block,
                once_marker='Labels are stored as long here'
            )
            cell['source'] = lines
            continue

        # 9) DataLoaders: document sampler and performance flags
        if 'train_loader = DataLoader' in src and 'val_loader' in src:
            lines = split_lines(src)
            block = [
                '# Use custom batch_sampler for balanced class batches on train;\n',
                '# pin_memory speeds host->device transfers on CUDA; num_workers for parallel loading.\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='train_loader =',
                insert_after=False,
                block_lines=block,
                once_marker='Use custom batch_sampler for balanced class batches'
            )
            cell['source'] = lines
            continue

        # 10) Eval/train loops: explain no_grad/eval/train, grads, updates, schedulers
        if 'def evaluate_epoch' in src and 'def train_epoch' in src:
            lines = split_lines(src)
            lines = ensure_comment_block(
                lines,
                anchor='@torch.no_grad',
                insert_after=True,
                block_lines=['# Inference mode: no gradients tracked and no parameter updates.\n'],
                once_marker='no gradients tracked and no parameter updates'
            )
            lines = ensure_comment_block(
                lines,
                anchor='ffn.eval()',
                insert_after=False,
                block_lines=['    # eval(): disable dropout/BN updates for deterministic evaluation\n'],
                once_marker='disable dropout/BN updates'
            )
            # train_epoch annotations
            lines = ensure_comment_block(
                lines,
                anchor='def train_epoch',
                insert_after=True,
                block_lines=['    # train(): enable dropout and other train-time layers\n', '    # emb_stack and ffn will accumulate grads from loss.backward()\n'],
                once_marker='emb_stack and ffn will accumulate grads'
            )
            lines = ensure_comment_block(
                lines,
                anchor='# forward',
                insert_after=True,
                block_lines=['        # Embedding lookups produce dense vectors; these receive gradients via backprop.\n'],
                once_marker='Embedding lookups produce dense vectors'
            )
            # Backward/step sequence
            back_block = [
                '        # 1) Zero grads on both optimizers\n',
                '        # 2) Backprop to compute grads for embeddings + all FFN params\n',
                '        # 3) Step optimizers to update weights; step schedulers to adjust LRs\n',
            ]
            lines = ensure_comment_block(
                lines,
                anchor='zero_grads',
                insert_after=False,
                block_lines=back_block,
                once_marker='Backprop to compute grads for embeddings'
            )
            cell['source'] = lines
            continue

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    annotate()

# Inference Prompts

Collection of prompts for testing model inference.

## Files

- `creative.txt` - Creative writing and storytelling prompts
- `technical.txt` - Programming and technical explanation prompts
- `knowledge.txt` - Factual and educational prompts
- `conversation.txt` - Chat-style conversational prompts
- `french.txt` - Prompts in French
- `benchmark.json` - Structured prompts for programmatic use

## Usage

### With inference.py

```bash
# Single prompt
python inference.py --checkpoint outputs/full/checkpoint_20000.pt \
    --prompt "Once upon a time, in a kingdom where dragons and humans lived in peace,"

# From file (one prompt per line)
python inference.py --checkpoint outputs/full/checkpoint_20000.pt \
    --prompt-file prompts/creative.txt --max-new-tokens 100
```

### Programmatic Usage

```python
import json

with open('prompts/benchmark.json') as f:
    prompts = json.load(f)

# Access by category
for prompt in prompts['creative']:
    print(prompt)
```

## Categories

| Category | Description | Use Case |
|----------|-------------|----------|
| creative | Story beginnings, fiction | Test narrative coherence |
| technical | Code, algorithms, explanations | Test technical knowledge |
| knowledge | Facts, history, science | Test factual accuracy |
| conversation | Chat format Q&A | Test instruction following |
| reasoning | Logic puzzles, math | Test reasoning ability |
| french | French language prompts | Test multilingual capability |

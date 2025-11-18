#!/usr/bin/env python
"""
Create properly formatted ChatML prompts for batch inference.
Each prompt should be on a single line with literal \n for newlines.
"""

prompts = [
    # With system prompt
    "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n<|im_start|>user\nWrite a short story about a robot learning to paint.\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>system\nYou are an expert programmer.\n<|im_end|>\n<|im_start|>user\nExplain the concept of recursion with a simple example.\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>system\nYou are a creative writing assistant.\n<|im_end|>\n<|im_start|>user\nDescribe a futuristic city where nature and technology coexist harmoniously.\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>system\nYou are a science educator.\n<|im_end|>\n<|im_start|>user\nExplain quantum entanglement in simple terms.\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>system\nYou are a philosophical thinker.\n<|im_end|>\n<|im_start|>user\nWhat is consciousness and can machines ever truly be conscious?\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>system\nYou are a helpful assistant specialized in machine learning.\n<|im_end|>\n<|im_start|>user\nCompare transformer models with recurrent neural networks for NLP tasks.\n<|im_end|>\n<|im_start|>assistant\n",

    # Without system prompt
    "<|im_start|>user\nWhat are the main differences between supervised and unsupervised learning?\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>user\nHow can we reduce carbon emissions in urban areas?\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>user\nWrite a Python function to find the nth Fibonacci number.\n<|im_end|>\n<|im_start|>assistant\n",

    "<|im_start|>user\nAnalyze the pros and cons of remote work for software developers.\n<|im_end|>\n<|im_start|>assistant\n",
]

# Write to file - each prompt on its own line
with open('prompts_chatml_fixed.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')

print(f"Created prompts_chatml_fixed.txt with {len(prompts)} ChatML formatted prompts")
print("\nFirst prompt preview:")
print(repr(prompts[0]))
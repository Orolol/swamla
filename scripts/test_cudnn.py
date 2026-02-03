import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# head_dim = 256
# ... définition de vos tenseurs Q, K, V ...

# On essaie de forcer cuDNN pour voir s'il accepte le kernel
try:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("Succès avec cuDNN sur large head_dim !")
except RuntimeError as e:
    print(f"Échec cuDNN : {e}")
    # Fallback probable vers FlashAttention
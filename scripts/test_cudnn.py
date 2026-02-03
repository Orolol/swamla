import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import sys

def check_system_info():
    print("=" * 50)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Version    : {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"Capacité Compute: {torch.cuda.get_device_capability(0)}")
        print(f"cuDNN Version   : {torch.backends.cudnn.version()}")
    else:
        print("❌ Erreur : Pas de GPU détecté.")
        sys.exit(1)
    print("=" * 50 + "\n")

def test_head_dim(head_dim, dtype=torch.bfloat16):
    # Configuration classique
    batch_size = 4
    num_heads = 8
    seq_len = 1024
    
    print(f"Testing head_dim = {head_dim} with dtype = {dtype}...")

    # Création des tenseurs sur GPU
    # Forme: (Batch, Heads, Seq_Len, Head_Dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # On force UNIQUEMENT le backend cuDNN.
    # Si cuDNN ne peut pas le gérer, cela doit planter.
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            # Le warmup n'est pas strictement nécessaire pour le crash test, 
            # mais bon pour initialiser les buffers
            _ = F.scaled_dot_product_attention(q, k, v)
            
            # Si on arrive ici, c'est que ça marche
            print(f"✅ SUCCÈS : cuDNN supporte head_dim={head_dim}")
            return True

    except RuntimeError as e:
        # Analyse de l'erreur pour comprendre pourquoi ça échoue
        err_msg = str(e)
        if "No execution plans support the graph" in err_msg:
            print(f"❌ ÉCHEC : cuDNN a rejeté le graphe (probablement dimension non supportée).")
        elif "CUDNN_ATTENTION is not available" in err_msg:
             print(f"❌ ÉCHEC : Le backend CUDNN n'est pas disponible sur cette build/hardware.")
        else:
            print(f"❌ ÉCHEC : Erreur Runtime : {err_msg}")
        return False
        
    except Exception as e:
        print(f"⚠️ Erreur inattendue : {e}")
        return False

def main():
    check_system_info()

    # Choix du dtype : BF16 est préféré sur Hopper/Ampere, FP16 sinon.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Utilisation de : {dtype}\n")

    # 1. Test de contrôle (Doit toujours fonctionner sur GPU moderne)
    print("--- Test de contrôle (Standard) ---")
    test_head_dim(128, dtype)
    print("-" * 30 + "\n")

    # 2. Test critique (Ce que tu veux savoir)
    print("--- Test Cible (Large Head) ---")
    test_head_dim(256, dtype)
    print("-" * 30 + "\n")
    
    # 3. Test exotique (Pour Hopper/Blackwell optimisé)
    print("--- Test Exotique (Ex: DeepSeek) ---")
    test_head_dim(192, dtype)

if __name__ == "__main__":
    main()
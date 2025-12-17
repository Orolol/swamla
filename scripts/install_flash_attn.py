#!/usr/bin/env python3
"""
Script pour installer automatiquement la bonne wheel de Flash Attention
en détectant les versions de CUDA, PyTorch et Python.
"""

import subprocess
import sys
import platform
import re


def get_python_version():
    """Retourne la version Python sous forme cpXXX (ex: cp312)."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"cp{major}{minor}"


def get_torch_version():
    """Retourne la version majeure.mineure de PyTorch (ex: 2.8)."""
    try:
        import torch
        version = torch.__version__
        # Extraire version majeure.mineure (ex: "2.8.0+cu124" -> "2.8")
        match = re.match(r"(\d+\.\d+)", version)
        if match:
            return match.group(1)
        return None
    except ImportError:
        return None


def get_cuda_version():
    """Retourne la version CUDA (ex: cu124, cu126)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Essayer d'obtenir depuis la version PyTorch d'abord
        torch_version = torch.__version__
        match = re.search(r"cu(\d+)", torch_version)
        if match:
            cuda_num = match.group(1)
            # Normaliser: cu124 -> cu124, cu126 -> cu126
            return f"cu{cuda_num[:2]}"  # On prend juste les 2 premiers chiffres (12 pour CUDA 12.x)

        # Sinon utiliser torch.version.cuda
        cuda_version = torch.version.cuda
        if cuda_version:
            # "12.4" -> "cu12"
            major = cuda_version.split('.')[0]
            return f"cu{major}"

        return None
    except ImportError:
        return None


def get_cxx11_abi():
    """Détecte si PyTorch utilise CXX11 ABI."""
    try:
        import torch
        # torch._C._GLIBCXX_USE_CXX11_ABI est True si CXX11 ABI est utilisé
        if hasattr(torch._C, '_GLIBCXX_USE_CXX11_ABI'):
            return torch._C._GLIBCXX_USE_CXX11_ABI
        # Par défaut, les versions récentes de PyTorch utilisent CXX11 ABI
        return True
    except:
        return True


def get_platform():
    """Retourne la plateforme (ex: linux_x86_64)."""
    system = platform.system().lower()
    machine = platform.machine()

    if system == "linux" and machine == "x86_64":
        return "linux_x86_64"
    elif system == "windows" and machine in ("AMD64", "x86_64"):
        return "win_amd64"
    else:
        return f"{system}_{machine}"


def get_available_flash_attn_versions():
    """Liste des versions de Flash Attention disponibles."""
    # Versions connues (à mettre à jour si nécessaire)
    return ["2.8.3", "2.8.2", "2.8.1", "2.8.0", "2.7.4", "2.7.3", "2.7.2", "2.7.1", "2.7.0"]


def build_wheel_url(flash_version, cuda_version, torch_version, python_version, cxx11_abi, platform_tag):
    """
    Construit l'URL de la wheel Flash Attention.

    Format: flash_attn-{version}+{cuda}torch{torch_ver}cxx11abi{ABI}-{python}-{python}-{platform}.whl
    Exemple: flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    """
    abi_str = "TRUE" if cxx11_abi else "FALSE"

    filename = f"flash_attn-{flash_version}+{cuda_version}torch{torch_version}cxx11abi{abi_str}-{python_version}-{python_version}-{platform_tag}.whl"
    url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{flash_version}/{filename}"

    return url, filename


def check_url_exists(url):
    """Vérifie si une URL existe (HTTP HEAD request)."""
    try:
        import urllib.request
        request = urllib.request.Request(url, method='HEAD')
        request.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except:
        return False


def install_wheel(url):
    """Installe une wheel depuis une URL."""
    print(f"\nInstallation de: {url}")
    result = subprocess.run([sys.executable, "-m", "pip", "install", url], capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Installation automatique de Flash Attention")
    print("=" * 60)

    # Détection des versions
    python_version = get_python_version()
    torch_version = get_torch_version()
    cuda_version = get_cuda_version()
    cxx11_abi = get_cxx11_abi()
    platform_tag = get_platform()

    print(f"\nVersions détectées:")
    print(f"  Python:    {python_version} ({sys.version.split()[0]})")
    print(f"  PyTorch:   {torch_version if torch_version else 'Non installé'}")
    print(f"  CUDA:      {cuda_version if cuda_version else 'Non disponible'}")
    print(f"  CXX11 ABI: {cxx11_abi}")
    print(f"  Platform:  {platform_tag}")

    # Vérifications
    if not torch_version:
        print("\nErreur: PyTorch n'est pas installé. Installez PyTorch d'abord.")
        sys.exit(1)

    if not cuda_version:
        print("\nErreur: CUDA n'est pas disponible. Flash Attention nécessite CUDA.")
        sys.exit(1)

    if platform_tag != "linux_x86_64":
        print(f"\nAvertissement: Plateforme {platform_tag} non officiellement supportée.")
        print("Les wheels pré-compilées sont principalement disponibles pour Linux x86_64.")

    # Recherche de la bonne wheel
    print("\nRecherche de la wheel compatible...")
    flash_versions = get_available_flash_attn_versions()

    found_url = None
    found_filename = None

    for flash_version in flash_versions:
        url, filename = build_wheel_url(
            flash_version, cuda_version, torch_version,
            python_version, cxx11_abi, platform_tag
        )

        print(f"  Tentative: {filename}...", end=" ")

        if check_url_exists(url):
            print("TROUVÉ!")
            found_url = url
            found_filename = filename
            break
        else:
            print("non disponible")

    if not found_url:
        print("\nAucune wheel pré-compilée trouvée pour votre configuration.")
        print("\nOptions:")
        print("  1. Compiler depuis les sources: pip install flash-attn --no-build-isolation")
        print("  2. Vérifier les releases: https://github.com/Dao-AILab/flash-attention/releases")
        print("\nConfiguration recherchée:")
        print(f"  - CUDA {cuda_version}, PyTorch {torch_version}, Python {python_version}")
        sys.exit(1)

    # Installation
    print(f"\nWheel trouvée: {found_filename}")

    response = input("\nInstaller maintenant? [O/n] ").strip().lower()
    if response in ("", "o", "oui", "y", "yes"):
        if install_wheel(found_url):
            print("\nFlash Attention installé avec succès!")

            # Vérification
            try:
                import flash_attn
                print(f"Version installée: {flash_attn.__version__}")
            except ImportError:
                print("Avertissement: Import de flash_attn a échoué.")
        else:
            print("\nErreur lors de l'installation.")
            sys.exit(1)
    else:
        print("\nInstallation annulée.")
        print(f"URL de la wheel: {found_url}")


if __name__ == "__main__":
    main()

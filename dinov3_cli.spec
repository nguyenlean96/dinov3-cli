# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ============================================================
# Collect all dependencies for ML packages
# ============================================================

# PyTorch - collect everything including CUDA (even if you don't use it, safer to include)
datas_torch, binaries_torch, hiddenimports_torch = collect_all('torch')
datas_torchvision, binaries_torchvision, hiddenimports_torchvision = collect_all('torchvision')

# Transformers - this is where it gets tricky
datas_transformers, binaries_transformers, hiddenimports_transformers = collect_all('transformers')

# HuggingFace Hub (transformers dependency)
datas_huggingface_hub, binaries_huggingface_hub, hiddenimports_huggingface_hub = collect_all('huggingface_hub')

# Additional transformers submodules that may be missed
hiddenimports_transformers_extra = collect_submodules('transformers')

# Safely handle optional dependencies
try:
    from PyInstaller.utils.hooks import collect_all as collect_all_optional
    datas_tokenizers, binaries_tokenizers, hiddenimports_tokenizers = collect_all_optional('tokenizers')
except Exception:
    datas_tokenizers, binaries_tokenizers, hiddenimports_tokenizers = [], [], []

# numpy is critical for torch
datas_numpy, binaries_numpy, hiddenimports_numpy = collect_all('numpy')

# Rich and Typer for CLI
datas_rich, binaries_rich, hiddenimports_rich = collect_all('rich')
datas_typer, binaries_typer, hiddenimports_typer = collect_all('typer')

# Combine all collected data
datas = (
    datas_torch + datas_torchvision + 
    datas_transformers + datas_huggingface_hub + 
    datas_tokenizers + datas_numpy + 
    datas_rich + datas_typer
)

binaries = (
    binaries_torch + binaries_torchvision + 
    binaries_transformers + binaries_huggingface_hub + 
    binaries_tokenizers + binaries_numpy + 
    binaries_rich + binaries_typer
)

hiddenimports = (
    hiddenimports_torch + hiddenimports_torchvision +
    hiddenimports_transformers + hiddenimports_transformers_extra +
    hiddenimports_huggingface_hub + hiddenimports_tokenizers +
    hiddenimports_numpy + hiddenimports_rich + hiddenimports_typer +
    [
        # Your package imports
        'dinov3_cli',
        'dinov3_cli.cli',
        'dinov3_cli.commands',
        'dinov3_cli.commands.extract',
        'dinov3_cli.core',
        'dinov3_cli.core.models',
        'dinov3_cli.core.processor',
        
        # Critical PyTorch JIT imports (frequently missed)
        'torch._C._nvrtc',
        'torch._C._jit',
        
        # matplotlib backend
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_pdf',
    ]
)

# ============================================================
# PyInstaller Analysis
# ============================================================

a = Analysis(
    ['bootstrap.py'],  # You'll create this
    pathex=['.'],
    hiddenimports=hiddenimports,
    binaries=binaries,
    datas=datas,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        'tkinter',
        'test',
        'pytest',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='dinov3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
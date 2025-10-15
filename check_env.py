import sys
import torch

def check_environment():
    print('\n=== Python and CUDA Environment ===')
    print(f'Python version: {sys.version}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU device: {torch.cuda.get_device_name(0)}')

    try:
        import transformers
        print(f'\n=== ML Libraries ===')
        print(f'Transformers version: {transformers.__version__}')
    except ImportError:
        print('Transformers not installed')

    try:
        import bitsandbytes
        print(f'Bitsandbytes version: {bitsandbytes.__version__}')
    except ImportError:
        print('Bitsandbytes not installed')

    try:
        import xformers
        print(f'xformers version: {xformers.__version__}')
    except ImportError:
        print('xformers not installed')

    try:
        import unsloth
        print(f'Unsloth version: {unsloth.__version__}')
    except ImportError:
        print('Unsloth not installed')

    try:
        import accelerate
        print(f'\n=== Additional Libraries ===')
        print(f'Accelerate version: {accelerate.__version__}')
    except ImportError:
        print('Accelerate not installed')

    try:
        import wandb
        print(f'Wandb version: {wandb.__version__}')
    except ImportError:
        print('Wandb not installed')

    try:
        import datasets
        print(f'Datasets version: {datasets.__version__}')
    except ImportError:
        print('Datasets not installed')

    try:
        import peft
        print(f'PEFT version: {peft.__version__}')
    except ImportError:
        print('PEFT not installed')

    try:
        import trl
        print(f'TRL version: {trl.__version__}')
    except ImportError:
        print('TRL not installed')

if __name__ == "__main__":
    check_environment()
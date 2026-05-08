import optuna
import torch


if __name__ == "__main__":
    print("Check that new package is working:")
    print(f"optuna.__version__ {optuna.__version__}")
    print("Check that PyTorch from container is still working:")
    print(f"torch.__version__ {torch.__version__}")
    print(f"torch.cuda.is_available() {torch.cuda.is_available()}")
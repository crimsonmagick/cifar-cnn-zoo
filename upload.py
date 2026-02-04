import os

import torch

from constants import CHECKPOINT_DIR
from model_services import model_for_training, model_from_checkpoint


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dirs = os.listdir(CHECKPOINT_DIR)
    for m_dir in model_dirs:
        rel_path = f'{CHECKPOINT_DIR}/{m_dir}'
        checkpoints = sorted(os.listdir(rel_path), reverse=True)
        if checkpoints:
            checkpoint_path = f"{rel_path}/{checkpoints[0]}"
            model, _ = model_from_checkpoint(checkpoint_path, device)







if __name__ == "__main__":
    main()

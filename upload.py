import json
import os

import torch
from huggingface_hub import upload_folder

from constants import CHECKPOINT_DIR, DIST_DIR
from model_services import model_from_checkpoint
from safetensors.torch import save_file


def generate_config(dist_model_path, model):
    config = {
        "model_name": model.model_name,
        "dataset": model.dataset.name.lower()
    }
    config_path = f"{dist_model_path}/config.json"
    with open(config_path, "w", encoding="utf-8", errors="strict") as f:
        json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dirs = os.listdir(CHECKPOINT_DIR)
    for m_dir in model_dirs:
        rel_path = f'{CHECKPOINT_DIR}/{m_dir}'
        checkpoints = sorted(os.listdir(rel_path), reverse=True)
        if checkpoints:
            checkpoint_path = f"{rel_path}/{checkpoints[0]}"
            model = model_from_checkpoint(checkpoint_path, device)
            print(f'{model.model_name} - {checkpoints[0]}')
            dist_model_path = f"{DIST_DIR}/{model.model_name}"
            os.makedirs(dist_model_path, exist_ok=True)
            save_file(model.state_dict(), f"{dist_model_path}/model.safetensors")
            generate_config(dist_model_path, model)
    upload_folder(
        repo_id="supermangoman/cifar-cnn-zoo",
        folder_path=DIST_DIR,
        commit_message="CIFAR model update"
    )


if __name__ == "__main__":
    main()

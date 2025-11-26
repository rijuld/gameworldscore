import os
import torch
from transformers import CLIPModel, CLIPProcessor
import requests

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Done.")

def main():
    output_dir = "models_for_rl_finetuning"
    os.makedirs(output_dir, exist_ok=True)

    # 1. CLIP (Temporal Consistency)
    print("Downloading CLIP (openai/clip-vit-base-patch32)...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.save_pretrained(os.path.join(output_dir, "clip-vit-base-patch32"))
        processor.save_pretrained(os.path.join(output_dir, "clip-vit-base-patch32"))
    except Exception as e:
        print(f"Failed to download CLIP: {e}")

    # 2. Aesthetic Predictor (Aesthetic Quality)
    aesthetic_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    download_file(aesthetic_url, os.path.join(output_dir, "aesthetic_predictor.pth"))

    # 3. IDM (Inverse Dynamics Model)
    print("Downloading IDM (4x_idm)...")
    idm_model_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model"
    idm_weights_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights"
    
    try:
        download_file(idm_model_url, os.path.join(output_dir, "4x_idm.model"))
        download_file(idm_weights_url, os.path.join(output_dir, "4x_idm.weights"))
    except Exception as e:
        print(f"Failed to download IDM: {e}")

if __name__ == "__main__":
    main()

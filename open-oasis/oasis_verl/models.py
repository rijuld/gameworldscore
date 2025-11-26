import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
import sys

# Add models_for_rl_finetuning to path (assuming it's in the parent directory of oasis_verl)
# This might need adjustment based on where this is run from.
# We assume the script is run from open-oasis/
sys.path.append(os.path.join(os.getcwd(), "models_for_rl_finetuning"))

try:
    from models_for_rl_finetuning.inverse_dynamics_model import IDMAgent
    from models_for_rl_finetuning.lib.actions import Buttons
    from utils import ACTION_KEYS # Assuming utils is in open-oasis/
except ImportError:
    print("Warning: Could not import IDMAgent or utils. Make sure models_for_rl_finetuning is set up and utils.py is available.")
    IDMAgent = None
    Buttons = None
    ACTION_KEYS = []

class MidasDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        self.transform = transform
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 360, 640)) # Return dummy image on error

class RewardModel(nn.Module):
    def __init__(self, device, w1=1.0, w2=1.0, w3=1.0):
        super().__init__()
        self.device = device
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        
        # 1. Load CLIP
        try:
            self.clip = CLIPModel.from_pretrained("models_for_rl_finetuning/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("models_for_rl_finetuning/clip-vit-base-patch32")
        except:
            print("Warning: Local CLIP not found, trying HuggingFace...")
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        # 2. Load Aesthetic Predictor
        self.aesthetic_head = nn.Linear(512, 1).to(device)
        aesthetic_path = "models_for_rl_finetuning/aesthetic_predictor.pth"
        if os.path.exists(aesthetic_path):
            try:
                state_dict = torch.load(aesthetic_path, map_location=device)
                if state_dict['weight'].shape[1] != 512:
                    print(f"Warning: Aesthetic weights have dim {state_dict['weight'].shape[1]}, but CLIP is 512. Using random weights.")
                else:
                    self.aesthetic_head.load_state_dict(state_dict)
            except Exception as e:
                print(f"Failed to load aesthetic weights: {e}")
        
        # 3. Load IDM (VPT)
        self.idm_agent = None
        idm_model_path = "models_for_rl_finetuning/4x_idm.model"
        idm_weights_path = "models_for_rl_finetuning/4x_idm.weights"
        
        if IDMAgent and os.path.exists(idm_model_path) and os.path.exists(idm_weights_path):
            try:
                agent_parameters = pickle.load(open(idm_model_path, "rb"))
                net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
                pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
                pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
                
                self.idm_agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=device)
                self.idm_agent.load_weights(idm_weights_path)
                print("Loaded VPT IDM.")
            except Exception as e:
                print(f"Failed to load VPT IDM: {e}")
        else:
            print("Warning: VPT IDM files not found or import failed. RIK reward will be 0.")

        # Freeze all reward models
        for p in self.parameters():
            p.requires_grad = False

    def reset(self):
        if self.idm_agent:
            self.idm_agent.reset()

    def compute_reward(self, frames, actions, t):
        if t < 1: return torch.tensor(0.0).to(self.device)

        curr_frame = frames[:, t]
        prev_frame = frames[:, t-1]
        
        # 1. Inverse Kinematics Score (RIK)
        rik = torch.tensor(0.0).to(self.device)
        if self.idm_agent:
            # Prepare input for IDM: (1, H, W, C) numpy array 0-255
            frame_np = (curr_frame[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame_batch = frame_np[None]
            
            # Get IDM prediction (logits/distribution)
            agent_input = self.idm_agent._video_obs_to_agent(frame_batch)
            dummy_first = torch.zeros((1, 1)).to(self.device)
            
            # Use policy.predict to get result with 'pd'
            with torch.no_grad():
                _, self.idm_agent.hidden_state, result = self.idm_agent.policy.predict(
                    agent_input, 
                    first=dummy_first, 
                    state_in=self.idm_agent.hidden_state,
                    deterministic=True
                )
            
            # result['pd'] is the distribution
            target_action_tensor = actions[:, t-1] # (1, 25)
            
            minerl_buttons = torch.zeros(1, len(Buttons.ALL)).to(self.device)
            minerl_camera = torch.zeros(1, 2).to(self.device)
            
            # Map buttons
            for i, key in enumerate(ACTION_KEYS):
                val = target_action_tensor[0, i]
                if key.startswith("camera"):
                    continue
                if key in Buttons.ALL:
                    idx = Buttons.ALL.index(key)
                    minerl_buttons[0, idx] = (val > 0.5).float()
            
            # Map camera
            cam_x_idx = ACTION_KEYS.index("cameraX")
            cam_y_idx = ACTION_KEYS.index("cameraY")
            cam_x_val = target_action_tensor[0, cam_x_idx]
            cam_y_val = target_action_tensor[0, cam_y_idx]
            
            vpt_cam_x = cam_x_val * 10.0
            vpt_cam_y = cam_y_val * 10.0
            
            vpt_cam_np = np.array([[vpt_cam_y.item(), vpt_cam_x.item()]]) # (1, 2)
            vpt_cam_bins = self.idm_agent.action_transformer.quantizer.discretize(vpt_cam_np)
            minerl_camera = torch.from_numpy(vpt_cam_bins).to(self.device)
            
            # Construct policy action dict
            policy_action = {
                "buttons": minerl_buttons.long().unsqueeze(1), # (1, 1, N_BUTTONS)
                "camera": minerl_camera.long().unsqueeze(1)    # (1, 1, 2)
            }
            
            # Compute log_prob
            log_prob = self.idm_agent.policy.pi_head.logprob(policy_action, result['pd'])
            
            rik = log_prob.sum() 

        # 2. Temporal Consistency (RTC)
        curr_224 = nn.functional.interpolate(curr_frame, size=(224, 224), mode='bilinear')
        prev_224 = nn.functional.interpolate(prev_frame, size=(224, 224), mode='bilinear')
        
        with torch.no_grad():
            curr_emb = self.clip.get_image_features(pixel_values=curr_224)
            prev_emb = self.clip.get_image_features(pixel_values=prev_224)
            
            curr_emb = curr_emb / curr_emb.norm(dim=-1, keepdim=True)
            prev_emb = prev_emb / prev_emb.norm(dim=-1, keepdim=True)
            
            rtc = (curr_emb * prev_emb).sum(dim=-1)

        # 3. Aesthetic Quality (RAQ)
        with torch.no_grad():
            raq = self.aesthetic_head(curr_emb).squeeze(-1)

        total_reward = self.w1 * rik + self.w2 * rtc + self.w3 * raq
        return total_reward

class ValueNetwork(nn.Module):
    def __init__(self, in_channels=16, hidden_dim=256):
        super().__init__()
        # Simple value network taking latents (B, C, H, W) and t (B, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 8)),
            nn.Flatten(),
            nn.Linear(hidden_dim * 5 * 8, 128), # Input 18x32 -> 9x16 -> 5x8
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, t):
        # x: (B, C, H, W)
        # t: (B, 1) - we can inject t later or ignore for now as a simple baseline
        return self.net(x)

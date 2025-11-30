"""
Minecraft gameplay dataset for Oasis RL finetuning.

Provides dataloaders for:
1. MinecraftDataset: Pre-recorded gameplay video for supervised training
2. MiDaSMinecraftDataset: MiDaS-60 image dataset with synthetic sequences
3. MinecraftRolloutDataset: On-policy rollout data for RL training
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.io import read_video, read_image
from torchvision.transforms.functional import resize
from torchvision import transforms
from einops import rearrange

from .action_utils import one_hot_actions, sample_random_action, ACTION_KEYS


class MiDaSMinecraftDataset(Dataset):
    """
    Dataset for MiDaS-60 Minecraft block images for RL finetuning.
    
    For RL training of world models, we only need:
    - **First frame**: The initial observation (from dataset)
    - **Action sequence**: Actions to condition generation on (randomly sampled)
    
    The world model (Oasis) will generate all subsequent frames.
    Rewards are computed on the GENERATED frames, not ground-truth.
    
    Expected structure:
        data_dir/
        ├── train/
        │   ├── acacia_log/
        │   │   ├── acacia_log_001.png
        │   │   └── ...
        │   └── ... (60 block categories)
        └── test/
            └── ... (same structure)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sequence_length: int = 32,
        frame_size: Tuple[int, int] = (360, 640),
        transform: Optional[callable] = None,
        sample_same_category: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing train/test folders
            split: 'train' or 'test'
            sequence_length: Length of action sequence to generate
            frame_size: Target frame size (H, W)
            transform: Optional transform to apply to first frame
            sample_same_category: Not used (kept for compatibility)
            augment: If True, apply data augmentation to first frame
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.transform = transform
        self.augment = augment
        
        # Set up augmentation transforms for the first frame
        if augment and split == "train":
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
            ])
        else:
            self.augment_transform = None
        
        # Index all images
        self.categories, self.images_by_category, self.all_images = self._index_images()
        
        print(f"MiDaSMinecraftDataset: Found {len(self.categories)} categories, "
              f"{len(self.all_images)} total images in {split}")
        print(f"  -> Each sample provides: 1 initial frame + {sequence_length} random actions")
    
    def _index_images(self) -> Tuple[List[str], Dict[str, List[Path]], List[Tuple[str, Path]]]:
        """Index all images by category."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            # Try without split subdirectory (flat structure)
            split_dir = self.data_dir
        
        categories = []
        images_by_category = {}
        all_images = []
        
        # Find all category directories
        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # Find all images in this category
            image_paths = sorted(list(category_dir.glob("*.png")) + 
                                list(category_dir.glob("*.jpg")) +
                                list(category_dir.glob("*.jpeg")))
            
            if len(image_paths) > 0:
                categories.append(category_name)
                images_by_category[category_name] = image_paths
                
                for img_path in image_paths:
                    all_images.append((category_name, img_path))
        
        return categories, images_by_category, all_images
    
    def __len__(self) -> int:
        # Each category can produce multiple sequences
        # Return a reasonable epoch size
        return len(self.all_images)
    
    def _load_and_preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        image = read_image(str(image_path))  # (C, H, W) uint8
        
        # Resize to target size
        image = resize(image, self.frame_size)
        
        # Convert to float [0, 1]
        image = image.float() / 255.0
        
        return image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single initial frame and action sequence for RL training.
        
        For RL finetuning of world models:
        - We provide ONLY the first frame as input
        - We provide a random action sequence
        - The world model generates all subsequent frames
        - Rewards are computed on generated frames (ground-truth free)
        
        Returns:
            Dict with:
                - initial_frame: (1, C, H, W) the first frame (prompt)
                - actions: (T, action_dim) random action sequence
                - category: category name (for logging)
        """
        # Get the image for this index
        category_name, image_path = self.all_images[idx]
        
        # Load and preprocess the initial frame
        initial_frame = self._load_and_preprocess_image(image_path)  # (C, H, W)
        
        # Apply augmentation to the initial frame
        if self.augment_transform is not None:
            initial_frame = self.augment_transform(initial_frame)
        
        # Apply custom transform if provided
        if self.transform is not None:
            initial_frame = self.transform(initial_frame)
        
        # Add time dimension: (C, H, W) -> (1, C, H, W)
        initial_frame = initial_frame.unsqueeze(0)
        
        # Generate random action sequence for world model conditioning
        # These actions will be used to generate the video
        actions = sample_random_action(self.sequence_length)
        
        return {
            'initial_frame': initial_frame,  # (1, C, H, W) - only the first frame!
            'actions': actions,  # (T, action_dim) - action sequence for generation
            'category': category_name,
        }


class MinecraftDataset(Dataset):
    """
    Dataset for pre-recorded Minecraft gameplay videos.
    
    Loads video clips and corresponding actions from the sample data format
    used by Oasis.
    
    Expected file format:
    - *.mp4: Video files
    - *.actions.pt or *.one_hot_actions.pt: Action files
    """
    
    def __init__(
        self,
        data_dir: str,
        clip_length: int = 32,
        frame_size: Tuple[int, int] = (360, 640),
        stride: int = 1,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_dir: Directory containing video and action files
            clip_length: Number of frames per clip
            frame_size: Target frame size (H, W)
            stride: Stride between consecutive clips
            transform: Optional transform to apply to frames
        """
        self.data_dir = Path(data_dir)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.transform = transform
        
        self.clips = self._index_clips()
    
    def _index_clips(self) -> List[Dict]:
        """Index all available video clips."""
        clips = []
        
        # Find all video files
        video_files = list(self.data_dir.glob("*.mp4"))
        
        for video_path in video_files:
            # Find corresponding action file
            base_name = video_path.stem
            actions_path = self.data_dir / f"{base_name}.one_hot_actions.pt"
            if not actions_path.exists():
                actions_path = self.data_dir / f"{base_name}.actions.pt"
            
            if not actions_path.exists():
                continue
            
            # Load video to get length
            video, _, _ = read_video(str(video_path), pts_unit="sec")
            num_frames = video.shape[0]
            
            # Create clips with stride
            for start_idx in range(0, num_frames - self.clip_length + 1, self.stride):
                clips.append({
                    'video_path': str(video_path),
                    'actions_path': str(actions_path),
                    'start_idx': start_idx,
                    'num_frames': num_frames,
                })
        
        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_info = self.clips[idx]
        
        # Load video clip
        video, _, _ = read_video(
            clip_info['video_path'],
            start_pts=clip_info['start_idx'],
            end_pts=clip_info['start_idx'] + self.clip_length,
            pts_unit="sec"
        )
        
        # Ensure we have correct number of frames
        if video.shape[0] < self.clip_length:
            # Pad with last frame
            pad = video[-1:].repeat(self.clip_length - video.shape[0], 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        video = video[:self.clip_length]
        
        # Resize and normalize
        video = video.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        video = torch.stack([
            resize(frame, self.frame_size) for frame in video
        ])
        
        # Load actions
        if clip_info['actions_path'].endswith('.actions.pt'):
            actions = torch.load(clip_info['actions_path'], weights_only=False)
            actions = one_hot_actions(actions)
        else:
            actions = torch.load(clip_info['actions_path'], weights_only=True)
        
        # Slice actions
        start = clip_info['start_idx']
        end = start + self.clip_length
        actions = actions[start:end]
        
        # Pad actions if needed
        if actions.shape[0] < self.clip_length:
            pad = torch.zeros(self.clip_length - actions.shape[0], actions.shape[1])
            actions = torch.cat([actions, pad], dim=0)
        
        if self.transform:
            video = self.transform(video)
        
        return {
            'frames': video,  # (T, C, H, W)
            'actions': actions,  # (T, action_dim)
        }


class MinecraftRolloutDataset(Dataset):
    """
    Dataset for on-policy rollout data in RL training.
    
    Stores rollout trajectories generated by the policy during training.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        device: str = "cpu",
    ):
        """
        Args:
            max_size: Maximum number of trajectories to store
            device: Device to store data on
        """
        self.max_size = max_size
        self.device = device
        
        self.trajectories: List[Dict] = []
    
    def add_trajectory(
        self,
        initial_frames: torch.Tensor,
        actions: torch.Tensor,
        generated_frames: torch.Tensor,
        rewards: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
    ):
        """
        Add a trajectory to the dataset.
        
        Args:
            initial_frames: (T_init, C, H, W) initial context frames
            actions: (T_gen, action_dim) actions taken
            generated_frames: (T_gen, C, H, W) generated frames
            rewards: (T_gen,) rewards received
            log_probs: Optional (T_gen,) log probabilities
        """
        trajectory = {
            'initial_frames': initial_frames.to(self.device),
            'actions': actions.to(self.device),
            'generated_frames': generated_frames.to(self.device),
            'rewards': rewards.to(self.device),
        }
        
        if log_probs is not None:
            trajectory['log_probs'] = log_probs.to(self.device)
        
        self.trajectories.append(trajectory)
        
        # Remove oldest if over capacity
        if len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size:]
    
    def clear(self):
        """Clear all trajectories."""
        self.trajectories = []
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.trajectories[idx]


def create_minecraft_dataloader(
    data_dir: str,
    batch_size: int = 4,
    clip_length: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    dataset_type: str = "auto",
    split: str = "train",
    frame_size: Tuple[int, int] = (360, 640),
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Minecraft data.
    
    Automatically detects dataset type based on directory structure.
    
    Args:
        data_dir: Directory containing data
        batch_size: Batch size
        clip_length: Number of frames per clip
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        dataset_type: 'auto', 'video', or 'midas'
        split: 'train' or 'test' (for MiDaS dataset)
        frame_size: Target frame size (H, W)
        
    Returns:
        DataLoader instance
    """
    data_path = Path(data_dir)
    
    # Auto-detect dataset type
    if dataset_type == "auto":
        # Check for MiDaS structure (train/test subdirs with category folders)
        if (data_path / "train").exists() or (data_path / "test").exists():
            dataset_type = "midas"
        # Check for video files
        elif len(list(data_path.glob("*.mp4"))) > 0:
            dataset_type = "video"
        # Check for category folders with images
        elif any((data_path / d).is_dir() and len(list((data_path / d).glob("*.png"))) > 0 
                 for d in os.listdir(data_path) if (data_path / d).is_dir()):
            dataset_type = "midas"
        else:
            raise ValueError(f"Could not detect dataset type in {data_dir}")
    
    # Extract DataLoader-specific kwargs before passing to dataset
    kwargs.pop('split', None)
    kwargs.pop('frame_size', None)
    pin_memory = kwargs.pop('pin_memory', True)
    prefetch_factor = kwargs.pop('prefetch_factor', 2)
    
    if dataset_type == "midas":
        dataset = MiDaSMinecraftDataset(
            data_dir=data_dir,
            split=split,
            sequence_length=clip_length,
            frame_size=frame_size,
            augment=(split == "train"),
            **kwargs,
        )
    else:
        dataset = MinecraftDataset(
            data_dir=data_dir,
            clip_length=clip_length,
            frame_size=frame_size,
            **kwargs,
        )
    
    # Custom collate function
    def collate_fn(batch):
        actions = torch.stack([item['actions'] for item in batch])
        
        # Handle both old format (frames) and new format (initial_frame)
        if 'initial_frame' in batch[0]:
            # New format: only first frame provided
            initial_frames = torch.stack([item['initial_frame'] for item in batch])
            result = {
                'initial_frame': initial_frames,  # (B, 1, C, H, W)
                'actions': actions,
            }
        else:
            # Old format: full sequence provided
            frames = torch.stack([item['frames'] for item in batch])
            result = {
                'frames': frames,
                'actions': actions,
            }
        
        # Include category if available
        if 'category' in batch[0]:
            result['category'] = [item['category'] for item in batch]
        
        return result
    
    persistent_workers = num_workers > 0  # Keep workers alive between batches
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def create_midas_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    sequence_length: int = 32,
    frame_size: Tuple[int, int] = (360, 640),
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for MiDaS dataset.
    
    Args:
        data_dir: Root directory containing MiDaS-60_small
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        frame_size: Target frame size (H, W)
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = MiDaSMinecraftDataset(
        data_dir=data_dir,
        split="train",
        sequence_length=sequence_length,
        frame_size=frame_size,
        augment=True,
    )
    
    test_dataset = MiDaSMinecraftDataset(
        data_dir=data_dir,
        split="test",
        sequence_length=sequence_length,
        frame_size=frame_size,
        augment=False,
    )
    
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        actions = torch.stack([item['actions'] for item in batch])
        categories = [item['category'] for item in batch]
        
        return {
            'frames': frames,
            'actions': actions,
            'category': categories,
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, test_loader


def load_prompt_and_actions(
    prompt_path: str,
    actions_path: Optional[str] = None,
    n_prompt_frames: int = 1,
    video_offset: Optional[int] = None,
    total_frames: int = 32,
    frame_size: Tuple[int, int] = (360, 640),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load prompt frames and actions for generation.
    
    Compatible with open-oasis/utils.py load functions.
    
    Args:
        prompt_path: Path to image or video
        actions_path: Path to action file (optional, generates random if None)
        n_prompt_frames: Number of prompt frames to use
        video_offset: Offset in video to start from
        total_frames: Total frames needed
        frame_size: Target frame size
        
    Returns:
        frames: (1, T, C, H, W) prompt frames
        actions: (1, T, action_dim) actions
    """
    IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
    VIDEO_EXTENSIONS = {"mp4"}
    
    ext = prompt_path.lower().split(".")[-1]
    
    if ext in IMAGE_EXTENSIONS:
        prompt = read_image(prompt_path)
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif ext in VIDEO_EXTENSIONS:
        prompt, _, _ = read_video(prompt_path, pts_unit="sec")
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
        prompt = prompt.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unknown file extension: {ext}")
    
    prompt = resize(prompt, frame_size)
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    
    # Load or generate actions
    if actions_path is not None:
        if actions_path.endswith(".actions.pt"):
            actions = one_hot_actions(torch.load(actions_path, weights_only=False))
        elif actions_path.endswith(".one_hot_actions.pt"):
            actions = torch.load(actions_path, weights_only=True)
        else:
            raise ValueError("Unknown action file extension")
        
        if video_offset is not None:
            actions = actions[video_offset:]
        
        actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
        actions = rearrange(actions[:total_frames], "t d -> 1 t d")
    else:
        # Generate random actions
        actions = sample_random_action(total_frames)
        actions = rearrange(actions, "t d -> 1 t d")
    
    return prompt, actions

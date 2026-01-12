import json
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import os


class NuScenesVLMDataset(Dataset):
    def __init__(self, jsonl_path, elements_path=None, data_root=None, image_transform=None, max_frames=None):
        self.samples = self._load_jsonl(Path(jsonl_path))
        self.data_root = data_root
        self.max_frames = max_frames
        self.skipped_count = 0  # 欠損クリップ数

        self.elements_data = {}
        if elements_path and os.path.exists(elements_path):
            with open(elements_path, "r") as f:
                self.elements_data = json.load(f)

        if image_transform is None:
            self.image_transform = transforms.ToTensor()
        else:
            self.image_transform = image_transform

    def _load_jsonl(self, path: Path):
        samples = []
        with open(path, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_id = sample["id"] # Assuming 'id' matches keys in elements json, or we use filename as key?
        # In refine_captions_llm.py, keys were from captions_inclusive.json. 
        # We need to ensure we can map sample -> elements.
        
        # Checking logic: keys in captions_inclusive were typically image filenames or clip IDs.
        # Let's assume for now we can find elements by trying image paths or ID.
        
        elements = []
        if self.elements_data:
            # Try to find elements using available keys
            # Strategy: refine_captions used keys from input json. 
            # If input json keys were filenames, we check sample['image_paths'][0]
            
            # Temporary fallback logic: check first image path
            # Image path in jsonl: /path/to/samples/CAM_FRONT/filename.jpg
            # Keys in elements.json: CAM_FRONT/filename.jpg
            full_path = sample["image_paths"][0]
            parts = full_path.split("/")
            if len(parts) >= 2:
                key_candidate = f"{parts[-2]}/{parts[-1]}"
            else:
                key_candidate = parts[-1]
            
            # Improved lookup:
            if key_candidate in self.elements_data:
                elements = self.elements_data[key_candidate]
            # Fallback: try just filename if above failed
            elif parts[-1] in self.elements_data:
                 elements = self.elements_data[parts[-1]]
            
            # If still not found, try stripping absolute path prefix if it somehow differs
            # But the above 2 cases cover 99% of scenarios.
            
            if not isinstance(elements, list):
                elements = [str(elements)]

        # Image loading logic remains for compatibility but is NOT used for Text-to-Text training
        # We can return None for images to save IO if training text-only
        
        return {
            "clip_id": clip_id,
            "text": sample["text"], # Original Dense Caption (Acts as Query)
            "elements": elements,   # Refined Elements (Acts as Positive Key)
            "meta": sample.get("meta", {}),
        }

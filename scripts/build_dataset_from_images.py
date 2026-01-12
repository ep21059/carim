
import os
import json
import uuid

def main():
    image_dir = "datasets/nuscenes_vlm/samples/CAM_FRONT"
    output_path = "datasets/nuscenes_vlm/processed/train.jsonl"
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    
    # 1. Scan Images
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found.")
        return

    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(files)} images in {image_dir}.")
    
    # 2. Build JSONL items
    abs_base = os.path.abspath(image_dir)
    
    for f in files:
        # Create a unique ID or use filename
        # We use parent/filename style key for consistency with future VLM output
        key = f"CAM_FRONT/{f}"
        
        item = {
            "id": key,
            "scene_name": "unknown", # Metadata lost
            "image_paths": [os.path.join(abs_base, f)],
            "text": "", # Placeholder, dense caption will come from VLM output later?
            # Actually, dataset.py usually expects 'text' to be the caption used for retrieval query.
            # But in CARIM training, we use ANI synthetics or VLM captions.
            # If we train "Text-to-Text", we need a "Ground Truth Description" for L_self.
            # Since we deleted captions, we HAVE NO TEXT yet.
            # We must generate VLM captions first, THEN populate this field?
            # OR we populate this field with the VLM output once available.
            # For now, we leave it empty or placeholder.
            "meta": {}
        }
        samples.append(item)
        
    # 3. Save
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Saved {len(samples)} items to {output_path}")

if __name__ == "__main__":
    main()

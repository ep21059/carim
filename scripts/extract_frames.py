import os
import cv2
from tqdm import tqdm
import argparse

def extract_frames(video_path, save_dir, fps=1):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Warning: FPS is 0 for {video_path}, skipping smart skipping.")
        skip_frame = 1 # Fallback
    else:
        skip_frame = int(video_fps / fps)
        if skip_frame < 1: skip_frame = 1

    count = 0
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % skip_frame == 0:
            frame_name = f"frame_{frame_id:05d}.jpg"
            save_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(save_path, frame)
            frame_id += 1
        
        count += 1
        
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()
    
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos.")
    
    for vid in tqdm(video_files):
        video_path = os.path.join(args.video_dir, vid)
        # Creating a subdirectory for each video
        video_name = os.path.splitext(vid)[0]
        save_dir = os.path.join(args.output_dir, video_name)
        
        extract_frames(video_path, save_dir, args.fps)

if __name__ == "__main__":
    main()

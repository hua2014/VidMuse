import argparse
from pathlib import Path
from demos.VidMuse_app import load_model, _do_predictions_for_get_video_emb

def infer(video_dir, output_dir):
    # Get all mp4 files in the directory
    video_files = list(Path(video_dir).glob('*.mp4'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file_path = Path(output_dir) / f"utt2videoemb_output.pt"
    if output_file_path.exists():
        print(f"Audio file already exists, skipping: {output_file_path}")
        return         
    for video_path in video_files:   
        video_embs = _do_predictions_for_get_video_emb(
            [str(video_path)], duration=30
        )
        print(f"Generated video file: {video_embs.shape}")

        # 需要参考inspiremusic的实现 将特征 写入 output_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidMuse inference script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    
    load_model(args.model_path)
    infer(args.video_dir, args.output_dir)
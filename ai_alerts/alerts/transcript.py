import os
# Force Hugging Face cache to the user's home folder to avoid permission errors on system folders
home_dir = os.path.expanduser("~")
os.environ["HF_HOME"] = os.path.join(home_dir, ".cache", "huggingface")

from .models import Transcript
import cv2
import torch
import traceback
from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image

import threading

# Configuration: Number of frames to sample from the clip for VLM analysis.
# 10-12 frames is a sweet spot for balance between temporal detail and speed on CPU.
NUM_SAMPLES = 10

_processor = None
_caption_model = None
_model_failed = False
_load_lock = threading.Lock()

def load_vlm():
    global _processor, _caption_model, _model_failed
    if _model_failed:
        return None, None

    # Double-checked locking to prevent race condition between threads
    if _processor is None or _caption_model is None:
        with _load_lock:
            if _processor is None or _caption_model is None:
                try:
                    caption_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                    print(f"[Transcript] Lazily loading modern video VLM '{caption_model_name}'...")
                    
                    _processor = AutoProcessor.from_pretrained(caption_model_name)
                    _caption_model = AutoModelForMultimodalLM.from_pretrained(
                        caption_model_name,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                    
                    print("[Transcript] Video VLM model loaded successfully.")
                except Exception as e:
                    print(f"[Transcript] FATAL ERROR: Could not load the Video VLM model.")
                    print(f"[Transcript] Details: {e}\n{traceback.format_exc()}")
                    _processor, _caption_model = None, None
                    _model_failed = True

    return _processor, _caption_model


def preload_vlm_async():
    """
    Spawns a background thread to load the VLM model if it is not already loaded.
    This runs concurrently while the video frames are being buffered.
    """
    global _processor, _caption_model, _model_failed
    if not _model_failed and (_processor is None or _caption_model is None):
        print("[Transcript] Preloading VLM model asynchronously in background thread...")
        thread = threading.Thread(target=load_vlm)
        thread.daemon = True
        thread.start()


def generate_transcript(alert, video_path_or_frames):
    if isinstance(video_path_or_frames, list):
        print(f"[Transcript] Processing {len(video_path_or_frames)} raw frames directly from memory...")
    else:
        print(f"[Transcript] Processing video from disk: {video_path_or_frames}")

    processor, caption_model = load_vlm()
    if caption_model is None:
        Transcript.objects.create(alert=alert, summary="Automated summary failed: AI model could not be loaded.")
        return

    frames = []

    if isinstance(video_path_or_frames, list):
        raw_frames = video_path_or_frames
        frame_count = len(raw_frames)
        if frame_count == 0:
            Transcript.objects.create(alert=alert, summary="No frames available in video.")
            return

        # Sample frames for inference
        num_samples = min(NUM_SAMPLES, frame_count)
        for i in range(num_samples):
            frame_index = i * frame_count // num_samples
            frame = raw_frames[frame_index]
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
    else:
        video_path = video_path_or_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            Transcript.objects.create(alert=alert, summary="No frames available in video.")
            return

        # Sample frames for inference
        num_samples = min(NUM_SAMPLES, frame_count)
        for i in range(num_samples):
            frame_index = i * frame_count // num_samples
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)

        cap.release()

    if not frames:
        Transcript.objects.create(alert=alert, summary="Failed to extract readable frames.")
        return

    try:
        # Ground prompt with the specific violation categories detected by YOLO to improve VLM quality
        violations_str = alert.violations if alert.violations else "safety violation"
        prompt_text = (
            f"This safety surveillance clip recorded a compliance violation: {violations_str}. "
            "Write a concise, professional safety report describing what the worker is doing, "
            "what they are wearing (and missing), and the nature of the safety violation. "
            "Do not write a frame-by-frame list or repeat sentences. Keep it under three sentences."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"}, 
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Use the chat template to format the prompt cleanly
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # SINGLE forward pass for the entire timeline, passing the frames as videos=[[frames]]
        inputs = processor(text=prompt, videos=[[frames]], return_tensors="pt").to(caption_model.device)
        
        with torch.no_grad():
            output_ids = caption_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Trim out the user prompt from the answer string
        generated_text = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        summary = f"Analysis of the event shows: {generated_text}"

    except Exception as e:
        print(f"[Transcript] FATAL ERROR during native video analysis.\nDetails: {e}\n{traceback.format_exc()}")
        summary = "A safety violation was recorded, but processing the timeline failed."

    Transcript.objects.create(alert=alert, summary=summary)
    print(f"[Transcript] Saved transcript for Alert {alert.id}: {summary}")
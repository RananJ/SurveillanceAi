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

_processor = None
_caption_model = None
_model_failed = False

def load_vlm():
    global _processor, _caption_model, _model_failed
    if _model_failed:
        return None, None

    if _processor is None or _caption_model is None:
        try:
            # Swap to the 500M model sitting in your cache
            caption_model_name = "HuggingFaceTB/SmolVLM2-500M-Instruct"
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


def generate_transcript(alert, video_path):
    print(f"[Transcript] Processing video: {video_path}")

    processor, caption_model = load_vlm()
    if caption_model is None:
        Transcript.objects.create(alert=alert, summary="Automated summary failed: AI model could not be loaded.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        Transcript.objects.create(alert=alert, summary="No frames available in video.")
        return

    # Keep sampling down to 16-20 frames for the 500M model to optimize processing cost
    num_samples = min(16, frame_count)
    frames = []

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
        # Build a single structural chat prompt asking for a precise chronological description
        messages = [
            {
                "role": "user",
                "content": [
                    # Include the video placeholder token
                    {"type": "video"}, 
                    {"type": "text", "text": "Describe what happens in this surveillance clip chronologically. Focus heavily on any accidents, safety violations, or unusual events."}
                ]
            }
        ]

        # Use the chat template to format the prompt cleanly
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # SINGLE forward pass for the entire timeline, passing the frames as videos=[[frames]]
        inputs = processor(text=prompt, videos=[[frames]], return_tensors="pt").to(caption_model.device)
        
        with torch.no_grad():
            output_ids = caption_model.generate(**inputs, max_new_tokens=100)
        
        # Trim out the user prompt from the answer string
        generated_text = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        summary = f"Analysis of the event shows: {generated_text}"

    except Exception as e:
        print(f"[Transcript] FATAL ERROR during native video analysis.\nDetails: {e}\n{traceback.format_exc()}")
        summary = "A safety violation was recorded, but processing the timeline failed."

    Transcript.objects.create(alert=alert, summary=summary)
    print(f"[Transcript] Saved transcript for Alert {alert.id}: {summary}")
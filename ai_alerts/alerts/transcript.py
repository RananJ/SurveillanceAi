from collections import Counter
import cv2
import torch
import traceback  # Import for detailed error logging
from transformers import AutoProcessor, AutoModelForCausalLM
from .models import Transcript
from PIL import Image

# Global variables for caching loaded models
_processor = None
_caption_model = None
_model_failed = False


def load_vlm():
    """
    Lazily load the VLM processor and model.
    Only runs when generate_transcript is called.
    """
    global _processor, _caption_model, _model_failed
    if _model_failed:
        return None, None

    if _processor is None or _caption_model is None:
        try:
            print("[Transcript] Lazily loading VLM model 'microsoft/git-base-vatex'...")
            caption_model_name = "microsoft/git-base-vatex"
            _processor = AutoProcessor.from_pretrained(caption_model_name)
            _caption_model = AutoModelForCausalLM.from_pretrained(
                caption_model_name
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            print("[Transcript] VLM model loaded successfully.")
        except Exception as e:
            print(f"[Transcript] FATAL ERROR: Could not load the VLM model. Transcripts will fail.")
            print(f"[Transcript] Details: {e}\n{traceback.format_exc()}")
            _processor = None
            _caption_model = None
            _model_failed = True

    return _processor, _caption_model


def generate_transcript(alert, video_path):
    print(f"[Transcript] Processing video: {video_path}")

    processor, caption_model = load_vlm()
    if caption_model is None:
        print("[Transcript] Cannot generate transcript, VLM model failed to load.")
        Transcript.objects.create(
            alert=alert, summary="Automated summary failed: AI model could not be loaded.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Transcript] Could not open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        Transcript.objects.create(
            alert=alert,
            summary="No frames available in video, unable to generate transcript."
        )
        return

    num_samples = min(30, frame_count)
    descriptions = []

    for i in range(num_samples):
        frame_index = i * frame_count // num_samples
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            # --- This is the part that is likely failing silently ---
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(
                caption_model.device)
            output_ids = caption_model.generate(**inputs, max_length=50)
            caption = processor.batch_decode(
                output_ids, skip_special_tokens=True)[0]
            descriptions.append(caption.lower())
        except Exception as e:
            # --- Now it will print a detailed error if it fails ---
            print(f"[Transcript] FATAL ERROR during frame captioning.")
            print(f"[Transcript] Details: {e}\n{traceback.format_exc()}")
            # We can continue to the next frame
            continue

    cap.release()

    counts = Counter(descriptions)
    most_common = counts.most_common(3)

    summary_parts = []
    for text, freq in most_common:
        summary_parts.append(f"'{text}' (observed in {freq} frames)")

    if not summary_parts:
        summary = "A safety violation was recorded, but no clear description could be generated."
    else:
        summary = "Analysis of the event shows: " + \
            ". ".join(summary_parts) + "."

    Transcript.objects.create(alert=alert, summary=summary)
    print(f"[Transcript] Summary: {summary}")
    print(f"[Transcript] Saved transcript for Alert {alert.id}")

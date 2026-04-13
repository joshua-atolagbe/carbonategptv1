"""
MiniGPT-Med — Batch Inference on a Folder of Images
=====================================================
Runs a single question against every image in a folder and saves results to CSV.

Usage
-----
    python minigptmed_batch_inference.py \
        --cfg-path  eval_configs/minigptv2_eval.yaml \
        --image-dir Med_examples_v2/ \
        --question  "Describe the findings in this medical image." \
        --output    results.csv \
        --gpu-id    0

Optional: pass multiple questions via a text file (one per line)
        --question-file questions.txt
"""

import argparse
import csv
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

# MiniGPT-Med imports
from minigpt4.common.config   import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

from minigpt4.datasets.builders import *   # noqa
from minigpt4.models             import *   # noqa
from minigpt4.processors         import *   # noqa
from minigpt4.runners            import *   # noqa
from minigpt4.tasks              import *   # noqa

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark     = False
cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-Med batch inference over a folder")
    parser.add_argument("--cfg-path",       default="eval_configs/minigptv2_eval.yaml")
    parser.add_argument("--image-dir",      required=True,
                        help="Folder containing medical images (searched recursively).")
    parser.add_argument("--question",       default="Describe this image in detail.",
                        help="Question to ask for every image.")
    parser.add_argument("--question-file",  default=None,
                        help="Optional .txt file with one question per line. "
                             "Each image will be queried with ALL questions.")
    parser.add_argument("--output",         default="results.csv",
                        help="Path for the output CSV file.")
    parser.add_argument("--gpu-id",         type=int,   default=0)
    parser.add_argument("--max-new-tokens", type=int,   default=300)
    parser.add_argument("--temperature",    type=float, default=0.6)
    parser.add_argument("--top-p",          type=float, default=0.9)
    parser.add_argument("--num-beams",      type=int,   default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--recursive",      action="store_true",
                        help="Search sub-folders recursively for images.")
    # Required by MiniGPT-Med's Config class internally — do not remove
    parser.add_argument("--options", nargs="+", default=None,
                        help="Override config values in key=value format (e.g. model.lora_r=32).")
    return parser.parse_args()


def collect_images(image_dir: str, recursive: bool) -> list[Path]:
    """Return sorted list of image paths in the directory."""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    glob_fn = root.rglob if recursive else root.glob
    images  = sorted(
        p for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return images


def build_model_and_chat(args, device):
    """Load config → model → vis_processor → Chat."""
    cfg = Config(args)

    print("[INFO] Loading model weights (this takes ~1–2 min on first run) …")
    model_config             = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model     = model_cls.from_config(model_config).to(device)
    model.eval()
    print("[INFO] Model ready ✓")

    vis_proc_cfg  = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = (
        registry.get_processor_class(vis_proc_cfg.name)
        .from_config(vis_proc_cfg)
    )

    chat = Chat(model, vis_processor, device=device)
    return chat


def make_conv_template() -> Conversation:
    return Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )


def infer_single(chat, image_path: Path, question: str,
                 conv_template: Conversation, gen_kwargs: dict) -> str:
    """Run inference for one (image, question) pair. Returns answer string."""
    # Step 1: load & preprocess image → normalised tensor [1, C, H, W]
    raw_image  = Image.open(image_path).convert("RGB")
    image_tensor = chat.vis_processor(raw_image).unsqueeze(0).to(chat.device)

    # Step 2: run ViT + projection → LLM embedding tensor
    # img_list must hold the projected embedding, NOT the raw image/array.
    # upload_img() does no encoding — it blindly appends whatever it receives,
    # so get_context_emb would crash calling .device on a PIL/numpy object.
    with torch.no_grad():
        image_emb, _ = chat.model.encode_img(image_tensor)   # [1, tokens, dim]

    # Step 3: build conversation state and populate img_list with the embedding
    chat_state = conv_template.copy()
    img_list   = [image_emb]                                  # tensor with .device ✓
    chat_state.append_message(chat_state.roles[0], "<Img><ImageHere></Img>")

    # Step 4: append the question and generate
    chat.ask(question, chat_state)

    answer, _ = chat.answer(
        conv      = chat_state,
        img_list  = img_list,
        **gen_kwargs,
    )
    return answer.strip()


def main():
    args   = parse_args()
    device = f"cuda:{args.gpu_id}" if (args.gpu_id >= 0 and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Device: {device}")

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(args.image_dir, args.recursive)
    if not images:
        raise RuntimeError(f"No images found in: {args.image_dir}")
    print(f"[INFO] Found {len(images)} image(s) in '{args.image_dir}'")

    # ── Collect questions ─────────────────────────────────────────────────────
    if args.question_file:
        with open(args.question_file) as f:
            questions = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(questions)} question(s) from {args.question_file}")
    else:
        questions = [args.question]

    # ── Build model ───────────────────────────────────────────────────────────
    chat          = build_model_and_chat(args, device)
    conv_template = make_conv_template()

    gen_kwargs = dict(
        max_new_tokens     = args.max_new_tokens,
        temperature        = args.temperature,
        top_p              = args.top_p,
        num_beams          = args.num_beams,
        repetition_penalty = args.repetition_penalty,
    )

    # ── Run inference & write CSV ─────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total   = len(images) * len(questions)
    success = 0
    errors  = 0

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "image_name", "question", "answer", "error"],
        )
        writer.writeheader()

        # tqdm gives a nice progress bar in the terminal
        with tqdm(total=total, desc="Inference", unit="query") as pbar:
            for img_path in images:
                for question in questions:
                    row = {
                        "image_path": str(img_path),
                        "image_name": img_path.name,
                        "question":   question,
                        "answer":     "",
                        "error":      "",
                    }
                    try:
                        answer       = infer_single(chat, img_path, question,
                                                    conv_template, gen_kwargs)
                        row["answer"] = answer
                        success += 1
                        # Also print to terminal
                        tqdm.write(f"\n📷 {img_path.name}")
                        tqdm.write(f"   Q: {question}")
                        tqdm.write(f"   A: {answer}")
                    except Exception as e:
                        row["error"] = str(e)
                        errors += 1
                        tqdm.write(f"\n❌ Error on {img_path.name}: {e}")
                        traceback.print_exc()

                    writer.writerow(row)
                    csv_file.flush()   # write immediately so progress isn't lost on crash
                    pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done!  ✅ {success} succeeded  ❌ {errors} failed")
    print(f"Results saved to: {output_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

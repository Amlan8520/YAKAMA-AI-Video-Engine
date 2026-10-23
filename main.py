#!/usr/bin/env python3
"""YAKAMA AI Video Engine.

A free AI video generator for Android + Google Colab using:
- edge-tts for free Hindi/English speech synthesis
- local Whisper for subtitle generation
- FFmpeg for 1080p MP4 rendering and segment merging
- resumable chunk processing for long-form scripts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    import edge_tts  # type: ignore
except Exception:  # pragma: no cover
    edge_tts = None

try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover
    whisper = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_TEMP_DIR = "temp"
DEFAULT_RESUME_FILE = "resume_state.json"
DEFAULT_SEGMENT_MINUTES = 10


def setup_logging(log_path: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not installed or not in PATH.")
    return ffmpeg


def resolve_background_video(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Background video not found: {path}")
    valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".gif"}
    if candidate.suffix.lower() not in valid_exts:
        raise ValueError(f"Unsupported background format: {candidate.suffix}")
    return str(candidate)


def detect_language(text: str) -> str:
    hi_chars = set("अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह")
    hi_count = sum(1 for ch in text if ch in hi_chars)
    en_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return "hi" if hi_count > en_count else "en"


def pick_voice(language: str) -> str:
    if language.lower() == "hi":
        return "hi-IN-SwaraNeural"
    return "en-US-AriaNeural"


def split_script_into_chunks(text: str, chunk_minutes: int = DEFAULT_SEGMENT_MINUTES) -> List[str]:
    """Partition a long script into chunks sized for roughly chunk_minutes of speech.

    This is an estimate; exact duration will vary based on speaking speed and voice.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The script is empty.")

    approx_chars_per_minute = 8000
    target_chars = max(3500, approx_chars_per_minute * max(1, chunk_minutes))

    paragraphs = raw.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        sentences = paragraph.split(". ")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and current_len + len(sentence) + 1 > target_chars:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    if not chunks:
        raise ValueError("No usable chunks were created from the script.")
    return chunks


def load_script_file(path: str) -> str:
    script_path = Path(path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")
    return script_path.read_text(encoding="utf-8")


def text_to_speech_chunk(text: str, output_path: str, language: str = "en") -> None:
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed. Install requirements.txt first.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    voice = pick_voice(language)
    logging.info("Generating TTS chunk with voice=%s, language=%s", voice, language)

    communicator = edge_tts.Communicate(text, voice=voice)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(communicator.save(str(output_file)))
    finally:
        loop.close()

    if not output_file.exists():
        raise RuntimeError(f"TTS output not created: {output_path}")


def generate_subtitles_srt(audio_path: str, subtitle_path: str, model_size: str = "base") -> None:
    if whisper is None:
        raise RuntimeError("Local Whisper is not installed. Install requirements.txt first.")
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install requirements.txt first.")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    segments = result.get("segments", [])
    lines: List[str] = []

    for idx, segment in enumerate(segments, start=1):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        lines.append(format_srt_line(idx, start, end, text))

    out_file = Path(subtitle_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Subtitle generation complete: %s", subtitle_path)


def format_srt_line(index: int, start: float, end: float, text: str) -> str:
    def fmt_time(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        secs, ms = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    return f"{index}\n{fmt_time(start)} --> {fmt_time(end)}\n{text}\n"


def run_ffmpeg_command(cmd: List[str], stage_name: str) -> None:
    logging.info("[%s] %s", stage_name, " ".join(cmd))
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed during {stage_name}: {message}")


def render_audio_to_1080p_video(audio_path: str, output_path: str, duration_seconds: Optional[float] = None) -> None:
    """Render a 1080p MP4 from an audio file using a black background.

    This keeps memory usage low and works well for long-form script rendering.
    """
    ffmpeg = ensure_ffmpeg()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        audio_path,
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1920x1080:d=20",
        "-shortest",
        "-map",
        "0:a:0",
        "-map",
        "1:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    if duration_seconds and duration_seconds > 0:
        # Some versions of FFmpeg support a duration override this way.
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            audio_path,
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=1920x1080:d={max(duration_seconds, 1)}",
            "-shortest",
            "-map",
            "0:a:0",
            "-map",
            "1:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

    run_ffmpeg_command(cmd, "render_audio_to_1080p_video")


def render_segment_with_background(audio_path: str, background_video: str, output_path: str) -> None:
    ffmpeg = ensure_ffmpeg()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        background_video,
        "-i",
        audio_path,
        "-filter_complex",
        "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v];[v][1:a]concat=n=1:v=1:a=1[vout][aout]",
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    run_ffmpeg_command(cmd, "render_segment_with_background")


def merge_video_segments(segment_paths: List[str], final_output: str) -> None:
    ffmpeg = ensure_ffmpeg()
    output_dir = Path(final_output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    concat_list = output_dir / "concat_list.txt"

    with open(concat_list, "w", encoding="utf-8") as handle:
        for segment in segment_paths:
            handle.write(f"file '{os.path.abspath(segment)}'\n")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        final_output,
    ]
    run_ffmpeg_command(cmd, "merge_video_segments")


def save_copy_to_drive(local_path: str, drive_dir: str) -> None:
    if not drive_dir:
        return
    destination_dir = Path(drive_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(local_path).name
    shutil.copy2(local_path, destination)
    logging.info("Saved output copy to Google Drive: %s", destination)


def load_resume_state(path: str) -> dict:
    state_file = Path(path)
    if not state_file.exists():
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        logging.warning("Resume state file was invalid JSON. Resetting state.")
        return {}


def save_resume_state(path: str, payload: dict) -> None:
    state_file = Path(path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_long_script(
    script_text: str,
    output_dir: str,
    language: Optional[str] = None,
    background_video: Optional[str] = None,
    chunk_minutes: int = DEFAULT_SEGMENT_MINUTES,
    drive_dir: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    selected_language = (language or detect_language(script_text)).lower()
    chunks = split_script_into_chunks(script_text, chunk_minutes=chunk_minutes)
    resume_path = str(Path(output_dir) / DEFAULT_RESUME_FILE)
    state = load_resume_state(resume_path)

    segment_files: List[str] = []
    segment_meta: List[dict] = []

    for idx, chunk in enumerate(chunks, start=1):
        key = f"segment_{idx}"
        chunk_dir = Path(output_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        wav_path = str(chunk_dir / f"segment_{idx:03d}.wav")
        mp4_path = str(chunk_dir / f"segment_{idx:03d}.mp4")
        srt_path = str(chunk_dir / f"segment_{idx:03d}.srt")

        if state.get(key):
            if Path(mp4_path).exists():
                segment_files.append(mp4_path)
                segment_meta.append({"index": idx, "mp4_path": mp4_path, "srt_path": srt_path})
            continue

        try:
            logging.info("Rendering chunk %s/%s", idx, len(chunks))
            text_to_speech_chunk(chunk, wav_path, language=selected_language)
            if background_video:
                render_segment_with_background(wav_path, background_video, mp4_path)
            else:
                render_audio_to_1080p_video(wav_path, mp4_path)
            generate_subtitles_srt(wav_path, srt_path)
            state[key] = True
            save_resume_state(resume_path, state)
            segment_files.append(mp4_path)
            segment_meta.append({"index": idx, "mp4_path": mp4_path, "srt_path": srt_path})
            logging.info("Chunk %s rendered successfully", idx)
        except Exception as exc:
            state[key] = False
            save_resume_state(resume_path, state)
            logging.exception("Chunk %s failed: %s", idx, exc)
            raise

    if not segment_files:
        raise RuntimeError("No video chunks were generated successfully.")

    final_path = str(Path(output_dir) / "final_video.mp4")
    final_srt = str(Path(output_dir) / "final_subtitles.srt")
    merge_video_segments(segment_files, final_path)

    with open(final_srt, "w", encoding="utf-8") as handle:
        for item in segment_meta:
            srt_file = item.get("srt_path")
            if not srt_file or not os.path.exists(srt_file):
                continue
            with open(srt_file, "r", encoding="utf-8") as srt_handle:
                handle.write(srt_handle.read())

    if drive_dir:
        save_copy_to_drive(final_path, drive_dir)
        save_copy_to_drive(final_srt, drive_dir)

    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YAKAMA AI Video Engine")
    parser.add_argument("--script", help="Text file containing the long-form script")
    parser.add_argument("--text", help="Raw text to render directly")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for rendered files")
    parser.add_argument("--language", choices=["en", "hi"], default=None, help="Optional override for the script language")
    parser.add_argument("--background-video", default=None, help="Optional MP4/MOV/MKV background video")
    parser.add_argument("--chunk-minutes", type=int, default=DEFAULT_SEGMENT_MINUTES, help="Target per-chunk duration in minutes")
    parser.add_argument("--drive-dir", default=None, help="Optional Google Drive folder path for copying output")
    parser.add_argument("--log-file", default="yakama.log", help="Log file path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_file)

    try:
        if args.script:
            script_text = load_script_file(args.script)
        elif args.text:
            script_text = args.text
        else:
            raise ValueError("Provide --script or --text")

        background_video = resolve_background_video(args.background_video)
        final_video = render_long_script(
            script_text,
            output_dir=args.output_dir,
            language=args.language,
            background_video=background_video,
            chunk_minutes=args.chunk_minutes,
            drive_dir=args.drive_dir,
        )
        logging.info("Final video generated at: %s", final_video)
    except Exception as exc:  # pragma: no cover
        logging.exception("Fatal render error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

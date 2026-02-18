#!/usr/bin/env python3
"""
Batch Dynamite Explosion Detector

- Scans a folder for WAV files
- Follows the same segmenting strategy as web_app_batch.py (5s segments for long files)
- Uses dynamite_detector.analyze_audio_for_explosion() as the detection model
- Builds a cross-file timeline of detections based on filename timestamps
- Saves outputs as CSV and JSON next to the input folder

Filename timestamp formats supported (same spirit as web_app_batch.py):
  - YYYYMMDD_HHMMSS (preferred)
  - YYYYMMDD (date only)
  - YYYY-MM-DD (date only)

If a file or segment has no timestamp in its name, the timeline entry will have null timestamp.
"""

import os
import re
import csv
import gc
import json
import argparse
import tempfile
import contextlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import librosa
import soundfile as sf
import pandas as pd

# Use the rule-based detector as the model
from dynamite_detector import analyze_audio_for_explosion, DETECTOR_CONFIG

# -----------------------------
# Config (tweakable via CLI)
# -----------------------------
SEGMENT_DURATION = 5           # seconds
SEGMENTATION_THRESHOLD = 10    # seconds; files longer than this are segmented
ALLOWED_EXTENSIONS = {"wav", "WAV"}

# -----------------------------
# Utils (aligned with web_app_batch.py)
# -----------------------------

def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    base = os.path.basename(filename)
    m = re.search(r"(\d{8})_(\d{6})", base)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_date_from_filename(filename: str) -> Optional[datetime.date]:
    pats = [r"(\d{8})_\d{6}", r"(\d{4})(\d{2})(\d{2})_", r"(\d{4})-(\d{2})-(\d{2})", r"(\d{8})"]
    base = os.path.basename(filename)
    for pat in pats:
        m = re.search(pat, base)
        if not m:
            continue
        try:
            if len(m.groups()) == 1:
                s = m.group(1)
                if len(s) == 8:
                    return datetime.strptime(s, "%Y%m%d").date()
            else:
                y, mo, d = m.groups()
                return datetime(int(y), int(mo), int(d)).date()
        except Exception:
            continue
    return None


@contextlib.contextmanager
def safe_audio_load(file_path):
    y = None
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        yield y, sr
    finally:
        if y is not None:
            del y
        gc.collect()


def get_audio_duration(file_path) -> Optional[float]:
    # try soundfile first
    try:
        with sf.SoundFile(file_path) as f:
            return len(f) / f.samplerate
    except Exception:
        try:
            with safe_audio_load(file_path) as (y, sr):
                return len(y) / sr if sr and y is not None else None
        except Exception:
            return None


def split_audio_file_5_seconds(file_path) -> List[str]:
    """Split into ~5s segments on disk, naming with segment-start timestamp if available.
    Returns list of segment paths (including the original path if splitting fails)."""
    segs: List[str] = []
    try:
        with safe_audio_load(file_path) as (y, sr):
            if y is None or sr is None:
                return [file_path]
            dur = len(y) / sr
            if dur < SEGMENT_DURATION:
                return [file_path]
            n = int(SEGMENT_DURATION * sr)
            tmpdir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            seg_idx = 0
            file_ts0 = extract_timestamp_from_filename(file_path)
            for i in range(0, len(y), n):
                chunk = y[i:i+n]
                if len(chunk) < sr * 2:  # skip tiny chunks
                    continue
                if file_ts0:
                    ts_seg = file_ts0 + timedelta(seconds=i // sr)
                    name = f"{ts_seg.strftime('%Y%m%d_%H%M%S')}_5s_{seg_idx:03d}.wav"
                else:
                    name = f"{base}_5s_{seg_idx:03d}.wav"
                out = os.path.join(tmpdir, name)
                try:
                    chunk_c = np.ascontiguousarray(chunk)
                    with sf.SoundFile(out, 'w', samplerate=sr, channels=1) as f:
                        f.write(chunk_c)
                    segs.append(out)
                    seg_idx += 1
                except Exception:
                    pass
                finally:
                    if 'chunk_c' in locals():
                        del chunk_c
        return segs if segs else [file_path]
    except Exception:
        return [file_path]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Core processing
# -----------------------------

def analyze_segment_or_file(path: str, original_filename: str, is_segment: bool,
                            segment_offset_s: float = 0.0) -> List[Dict[str, Any]]:
    """Run the dynamite detector on a path. Returns a list of timeline rows (0 or 1 per detection)."""
    res = analyze_audio_for_explosion(path, DETECTOR_CONFIG)
    rows: List[Dict[str, Any]] = []

    # Only add to timeline if an explosion was detected
    if res.get('is_explosion'):
        # Explosion time is relative to the audio we analyzed (segment or file)
        t_rel = float(res.get('explosion_time_s', 0.0))

        # Exact second in the original file
        explosion_second_in_original = segment_offset_s + t_rel

        # Derive absolute time using filename timestamp
        ts_file = extract_timestamp_from_filename(os.path.basename(path if is_segment else original_filename))
        abs_ts: Optional[datetime] = None
        if ts_file:
            abs_ts = ts_file + timedelta(seconds=t_rel)

        # Derive a simple confidence proxy from available metrics if present
        metrics = res.get('metrics', {}) or {}
        amp_ratio = float(metrics.get('amplitude_ratio', 1.0)) if isinstance(metrics, dict) else 1.0
        confidence = float(np.clip(amp_ratio, 0.2, 1.0))  # coarse proxy since model is rule-based

        rows.append({
            'filename': os.path.basename(original_filename),
            'segment_path': os.path.basename(path) if is_segment else None,
            'detected': True,
            'confidence': confidence,
            'explosion_second_in_original_file': round(explosion_second_in_original, 2),
            'timestamp': abs_ts.isoformat() if abs_ts else None,
            'timestamp_ms': int(abs_ts.timestamp() * 1000) if abs_ts else None,
            'date': abs_ts.strftime('%Y%m%d') if abs_ts else (extract_date_from_filename(original_filename).strftime('%Y%m%d') if extract_date_from_filename(original_filename) else 'unknown'),
            'processing_method': 'segmented' if is_segment else 'whole_file',
            'explosion_time_s_in_analyzed_audio': t_rel
        })
    return rows


def process_folder(input_dir: str, recursive: bool = False,
                   seg_threshold: int = SEGMENTATION_THRESHOLD,
                   segment_duration: int = SEGMENT_DURATION) -> Dict[str, Any]:
    global SEGMENT_DURATION, SEGMENTATION_THRESHOLD
    SEGMENT_DURATION = int(segment_duration)
    SEGMENTATION_THRESHOLD = int(seg_threshold)

    timeline: List[Dict[str, Any]] = []
    processed_files = 0
    detections = 0

    walk_iter = os.walk(input_dir) if recursive else [(input_dir, [], os.listdir(input_dir))]

    for root, _dirs, files in walk_iter:
        wavs = [f for f in files if allowed_file(f)]
        for fn in sorted(wavs):
            fp = os.path.join(root, fn)
            processed_files += 1
            try:
                dur = get_audio_duration(fp) or 0.0
                if dur > SEGMENTATION_THRESHOLD:
                    segs = split_audio_file_5_seconds(fp)
                    for seg_i, sp in enumerate(segs):
                        seg_offset = seg_i * SEGMENT_DURATION
                        rows = analyze_segment_or_file(sp, original_filename=fn, is_segment=True,
                                                       segment_offset_s=seg_offset)
                        if rows:
                            detections += len(rows)
                            timeline.extend(rows)
                    # cleanup temporary segments (keep original)
                    for sp in segs:
                        if sp != fp:
                            try:
                                os.remove(sp)
                            except Exception:
                                pass
                else:
                    rows = analyze_segment_or_file(fp, original_filename=fn, is_segment=False)
                    if rows:
                        detections += len(rows)
                        timeline.extend(rows)
            except Exception:
                pass
            finally:
                gc.collect()

    # Sort timeline by timestamp if available
    timeline_sorted = sorted(
        timeline,
        key=lambda r: (r['timestamp_ms'] is None, r.get('timestamp_ms', 0))
    )

    summary = {
        'folder': os.path.abspath(input_dir),
        'processed_files': processed_files,
        'detections': detections,
        'segmentation_threshold_s': SEGMENTATION_THRESHOLD,
        'segment_duration_s': SEGMENT_DURATION,
        'generated_at': datetime.now().isoformat(),
    }

    return {
        'summary': summary,
        'timeline': timeline_sorted,
    }


MIN_CONFIDENCE = 0.9  # Only keep detections above this threshold


def write_outputs(result: Dict[str, Any], output_prefix: str, output_dir: str,
                  min_confidence: float = MIN_CONFIDENCE) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{output_prefix}.json")
    xlsx_path = os.path.join(output_dir, f"{output_prefix}.xlsx")

    # Filter timeline by confidence
    filtered = [r for r in result['timeline'] if r.get('confidence', 0) >= min_confidence]
    result_filtered = {**result, 'timeline': filtered}
    result_filtered['summary'] = {**result['summary'],
                                   'detections_after_filter': len(filtered),
                                   'min_confidence': min_confidence}

    # JSON
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(result_filtered, jf, ensure_ascii=False, indent=2)

    # Excel
    if filtered:
        df = pd.DataFrame(filtered)
        col_order = [
            'filename', 'confidence', 'explosion_second_in_original_file',
            'timestamp', 'date', 'segment_path', 'processing_method',
            'explosion_time_s_in_analyzed_audio'
        ]
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
    else:
        df = pd.DataFrame(columns=[
            'filename', 'confidence', 'explosion_second_in_original_file',
            'timestamp', 'date'
        ])
    df.to_excel(xlsx_path, index=False, engine='openpyxl')

    print(f'\nDetections before filter: {len(result["timeline"])}')
    print(f'Detections after filter (confidence >= {min_confidence}): {len(filtered)}')

    return {'json': json_path, 'xlsx': xlsx_path}


def main():
    p = argparse.ArgumentParser(
        description='Batch dynamite explosion detector with timeline output (uses dynamite_detector)'
    )
    p.add_argument('input_dir', help='Folder containing WAV files')
    p.add_argument('--recursive', action='store_true', help='Search subfolders recursively')
    p.add_argument('--seg-threshold', type=int, default=SEGMENTATION_THRESHOLD, help='Seconds; files longer than this are segmented')
    p.add_argument('--segment-dur', type=int, default=SEGMENT_DURATION, help='Segment length in seconds')
    p.add_argument('--output-prefix', default='explosion_timeline', help='Base name for output files (no extension)')
    p.add_argument('--output-dir', default='.', help='Directory to save outputs (default: current)')

    args = p.parse_args()

    result = process_folder(
        input_dir=args.input_dir,
        recursive=args.recursive,
        seg_threshold=args.seg_threshold,
        segment_duration=args.segment_dur,
    )

    paths = write_outputs(result, output_prefix=args.output_prefix, output_dir=args.output_dir)

    print("\nSummary:")
    for k, v in result['summary'].items():
        print(f"  {k}: {v}")
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()

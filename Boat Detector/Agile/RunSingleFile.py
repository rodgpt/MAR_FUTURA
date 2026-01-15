import argparse
import csv
import os
import sys
import tempfile

from perch_hoplite.agile import classifier as agile_classifier
from perch_hoplite.agile import embed
from perch_hoplite.agile import colab_utils
from perch_hoplite.agile import source_info
from perch_hoplite.agile.classifier import LinearClassifier


WAV_PATH = "/Volumes/Card3a/20251211_132600.WAV"
DB_PATH = "/Users/Rodrigo/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/Hydrophones/BOAT DETECTOR AGILE/single_file_db"
CLASSIFIER_PATH = "/Users/Rodrigo/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/Hydrophones/BOAT DETECTOR AGILE/agile_classifier_v2.pt"
MODEL_CHOICE = "perch_8"
LOGIT_THRESHOLD = 1.25
DATASET_NAME = "SingleFileDataset"
MIN_AUDIO_LEN_S = 1.0
USE_FILE_SHARDING = True
SHARD_LENGTH_IN_SECONDS = 5.0
AUDIO_WORKER_THREADS = 2
EMBED_BATCH_SIZE = 8
LABEL = "boat"
DEBUG_PRINT_INFERENCE_CSV_HEAD = True
DEBUG_DUMP_ALL_WINDOWS = True
MIN_SEGMENTS_OVER_THRESHOLD = 2
MIN_CONSECUTIVE_SEGMENTS = 2


def _float_or_none(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bool_decision_from_csv(csv_path: str, target_label: str, threshold: float) -> tuple[bool, float | None, int]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return False, None, 0

    # Filter by label if present.
    if reader.fieldnames and "label" in reader.fieldnames:
        rows = [r for r in rows if str(r.get("label", "")) == str(target_label)]

    if not rows:
        return False, None, 0

    # Prefer logit if present, otherwise fall back to score/prob if present.
    score_col = None
    if reader.fieldnames:
        for c in ("logit", "logits", "score", "prob", "probability"):
            if c in reader.fieldnames:
                score_col = c
                break

    if score_col is None:
        # If we can't find a score column, we still report detections based on row count.
        return True, None, int(len(rows))

    scores = [_float_or_none(r.get(score_col)) for r in rows]
    scores = [s for s in scores if s is not None]
    if not scores:
        return False, None, 0

    best = float(max(scores))
    return best >= threshold, best, sum(1 for s in scores if s >= threshold)


def _longest_consecutive_run(values: list[float], step: float) -> int:
    if not values:
        return 0

    values = sorted(set(values))
    step = float(step)

    # Tolerate small floating-point representation errors.
    eps = max(1e-6, abs(step) * 1e-6)

    best = 1
    current = 1
    for i in range(1, len(values)):
        if abs((values[i] - values[i - 1]) - step) <= eps:
            current += 1
        else:
            best = max(best, current)
            current = 1
    best = max(best, current)
    return best


def _score_summary_from_csv(
    csv_path: str,
    target_label: str,
    threshold: float,
) -> tuple[float | None, int, list[float]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, 0, []

    # Filter by label if present.
    if reader.fieldnames and "label" in reader.fieldnames:
        rows = [r for r in rows if str(r.get("label", "")) == str(target_label)]

    if not rows:
        return None, 0, []

    score_col = None
    if reader.fieldnames:
        for c in ("logit", "logits", "score", "prob", "probability"):
            if c in reader.fieldnames:
                score_col = c
                break

    if not score_col:
        return None, 0, []

    best_score: float | None = None
    starts_over: list[float] = []
    count_over = 0

    for r in rows:
        s = _float_or_none(r.get(score_col))
        if s is None:
            continue
        if best_score is None or s > best_score:
            best_score = float(s)
        if s >= threshold:
            count_over += 1
            ws = _float_or_none(r.get("window_start"))
            if ws is not None:
                starts_over.append(float(ws))

    return best_score, count_over, starts_over


def _print_label_summary(csv_path: str) -> None:
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print("Could not read inference CSV for summary:", repr(e))
        return

    if not rows:
        print("inference_csv_rows: 0")
        return

    score_col = None
    if reader.fieldnames:
        for c in ("logit", "logits", "score", "prob", "probability"):
            if c in reader.fieldnames:
                score_col = c
                break

    labels = sorted({str(r.get("label", "")) for r in rows})
    print("labels_in_csv:", labels)
    print("score_column:", score_col)

    if not score_col:
        return

    per_label_best: dict[str, float] = {}
    for r in rows:
        lbl = str(r.get("label", ""))
        s = _float_or_none(r.get(score_col))
        if s is None:
            continue
        if lbl not in per_label_best or s > per_label_best[lbl]:
            per_label_best[lbl] = s

    if per_label_best:
        print("best_score_per_label:")
        for k in sorted(per_label_best.keys()):
            print(f"  {k}: {per_label_best[k]:.4f}")


def _build_args_from_defaults() -> argparse.Namespace:
    return argparse.Namespace(
        wav_path=WAV_PATH,
        db_path=DB_PATH,
        classifier_path=CLASSIFIER_PATH,
        model_choice=MODEL_CHOICE,
        dataset_name=DATASET_NAME,
        min_audio_len_s=float(MIN_AUDIO_LEN_S),
        use_file_sharding=bool(USE_FILE_SHARDING),
        shard_length_in_seconds=float(SHARD_LENGTH_IN_SECONDS),
        audio_worker_threads=int(AUDIO_WORKER_THREADS),
        embed_batch_size=int(EMBED_BATCH_SIZE),
        label=LABEL,
        logit_threshold=float(LOGIT_THRESHOLD),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AGILE boat/not-boat inference for a single WAV file.")
    parser.add_argument("wav_path", help="Path to a .wav/.WAV file")
    parser.add_argument("--db_path", required=True, help="Folder where the Hoplite DB will be created/read")
    parser.add_argument("--classifier_path", required=True, help="Path to the trained AGILE LinearClassifier")
    parser.add_argument("--model_choice", default="perch_8", help="Embedding model key (must match training)")
    parser.add_argument("--dataset_name", default="SingleFileDataset")
    parser.add_argument("--min_audio_len_s", type=float, default=1.0)
    parser.add_argument("--use_file_sharding", action="store_true", default=True)
    parser.add_argument("--no_file_sharding", dest="use_file_sharding", action="store_false")
    parser.add_argument("--shard_length_in_seconds", type=float, default=5.0)
    parser.add_argument("--audio_worker_threads", type=int, default=2)
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--label", default="boat", help="Label to decide on")
    parser.add_argument("--logit_threshold", type=float, default=2.0)

    # If you run this script without CLI args, it will use the constants at the
    # top of this file.
    if len(sys.argv) <= 1:
        args = _build_args_from_defaults()
    else:
        args = parser.parse_args()

    wav_path = os.path.abspath(args.wav_path)
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if not args.db_path:
        raise ValueError("db_path is empty. Set DB_PATH at the top of the file or pass --db_path.")

    if not args.classifier_path or not os.path.exists(args.classifier_path):
        raise FileNotFoundError(f"Classifier not found: {args.classifier_path}")

    os.makedirs(args.db_path, exist_ok=True)

    base_dir = os.path.dirname(wav_path)
    file_glob = os.path.basename(wav_path)

    shard_len_s = float(args.shard_length_in_seconds) if args.use_file_sharding else None
    print("wav_path:", wav_path)
    print("use_file_sharding:", bool(args.use_file_sharding))
    print("shard_len_s:", shard_len_s)

    try:
        import soundfile as sf

        info = sf.info(wav_path)
        duration_s = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        if shard_len_s and duration_s > 0:
            expected_shards = int(duration_s // shard_len_s)
            if duration_s % shard_len_s != 0:
                expected_shards += 1
        else:
            expected_shards = 1
        print(f"audio_duration_s: {duration_s:.2f}")
        print("expected_shards (approx):", expected_shards)
    except Exception as e:
        print("Could not read audio duration via soundfile:", repr(e))

    audio_glob = source_info.AudioSourceConfig(
        dataset_name=args.dataset_name,
        base_path=base_dir,
        file_glob=file_glob,
        min_audio_len_s=float(args.min_audio_len_s),
        target_sample_rate_hz=-2,
        shard_len_s=shard_len_s,
    )

    configs = colab_utils.load_configs(
        source_info.AudioSources((audio_glob,)),
        args.db_path,
        model_config_key=args.model_choice,
        db_key="sqlite_usearch",
    )

    db = configs.db_config.load_db()

    before = db.count_embeddings()
    print("embeddings_in_db_before:", before)

    worker = embed.EmbedWorker(
        audio_sources=configs.audio_sources_config,
        db=db,
        model_config=configs.model_config,
        audio_worker_threads=int(args.audio_worker_threads),
    )

    # Embed the single file (and its shards, if sharding is enabled).
    worker.process_all(target_dataset_name=audio_glob.dataset_name, batch_size=int(args.embed_batch_size))

    after = db.count_embeddings()
    print("embeddings_in_db_after:", after)
    print("new_embeddings_added:", after - before)

    linear_classifier = LinearClassifier.load(args.classifier_path)

    with tempfile.NamedTemporaryFile(prefix="agile_single_", suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name

    threshold_for_csv = float(args.logit_threshold)
    if DEBUG_DUMP_ALL_WINDOWS:
        threshold_for_csv = -1.0e9

    agile_classifier.write_inference_csv(
        linear_classifier,
        db,
        tmp_csv,
        threshold_for_csv,
        labels=None,
    )

    if DEBUG_PRINT_INFERENCE_CSV_HEAD:
        try:
            with open(tmp_csv, "r", newline="") as f:
                for i, line in enumerate(f):
                    print(line.rstrip("\n"))
                    if i >= 10:
                        break
        except Exception as e:
            print("Could not print inference CSV head:", repr(e))

    _print_label_summary(tmp_csv)

    best_score, count_over, starts_over = _score_summary_from_csv(
        tmp_csv,
        args.label,
        float(args.logit_threshold),
    )

    longest_run = _longest_consecutive_run(starts_over, float(args.shard_length_in_seconds))

    is_boat = (count_over >= int(MIN_SEGMENTS_OVER_THRESHOLD)) and (longest_run >= int(MIN_CONSECUTIVE_SEGMENTS))

    verdict = "boat" if is_boat else "not_boat"
    print("THIS IS A BOAT" if is_boat else "THIS IS NOT A BOAT")
    if best_score is None:
        print(
            f"{verdict} (segments_over_threshold={count_over}, min_segments={MIN_SEGMENTS_OVER_THRESHOLD}, longest_consecutive={longest_run}, min_consecutive={MIN_CONSECUTIVE_SEGMENTS}, threshold={args.logit_threshold})"
        )
    else:
        print(
            f"{verdict} (best_score={best_score:.4f}, segments_over_threshold={count_over}, min_segments={MIN_SEGMENTS_OVER_THRESHOLD}, longest_consecutive={longest_run}, min_consecutive={MIN_CONSECUTIVE_SEGMENTS}, threshold={args.logit_threshold})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

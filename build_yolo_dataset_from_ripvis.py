from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable

from PIL import Image
from tqdm import tqdm


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RIPVIS_FPS_VERSION = "v1.8.4"


def ripvis_fps_v184() -> dict[str, float]:
    """
    Hardcoded RipVIS FPS table for RipVISv1.8.4 (video_id -> fps).

    Note: 'NR' videos reuse the same FPS as their corresponding RipVIS-<id>.
    """

    # Default in the dataset is mostly ~29.97, with a set of known exceptions.
    fps: dict[str, float] = {f"{i:03d}": 29.97 for i in range(1, 151)}
    fps.update(
        {
            "004": 24.91,
            "007": 25.0,
            "008": 25.0,
            "010": 25.0,
            "013": 29.79,
            "043": 29.98,
            "045": 29.98,
            "065": 23.98,
            "066": 23.98,
            "076": 30.0,
            "077": 30.0,
            "078": 30.0,
            "079": 30.0,
            "080": 30.0,
            "086": 59.94,
            "087": 59.95,
            "088": 59.95,
            "089": 59.95,
            "090": 59.95,
            "091": 29.96,
            "092": 29.96,
            "093": 29.96,
            "094": 29.96,
            "095": 30.01,
            "096": 30.01,
            "097": 30.01,
            "098": 30.01,
            "099": 30.01,
            "100": 29.98,
            "101": 30.0,
            "102": 30.0,
            "103": 30.0,
            "104": 30.0,
            "105": 30.0,
            "106": 30.0,
            "107": 30.0,
            "108": 30.0,
            "109": 30.0,
            "111": 59.94,
            "112": 59.94,
            "113": 59.94,
            "114": 59.94,
            "115": 59.94,
            "116": 29.96,
            "119": 30.0,
            "123": 30.0,
            "124": 30.0,
            "125": 23.976,
            "126": 50.0,
            "127": 25.0,
            "130": 23.976,
            "131": 23.976,
            "132": 25.0,
            "134": 59.94,
            "135": 24.0,
            "136": 23.976,
            "140": 23.976,
            "141": 25.0,
            "143": 59.94,
            "144": 59.94,
            "145": 59.94,
            "147": 25.0,
        }
    )
    return fps


@dataclass(frozen=True)
class VideoInfo:
    video_key: str  # e.g. "P-003" or "N-002"
    bucket: str  # "positive" | "negative"
    video_id: str  # "003" (no P-/N- prefix)
    orientation: str  # "H" | "V"
    difficulty: str | None  # "Easy" | "Medium" | "Hard" for positive, else None
    score: float | None  # video mean score for positive, else None
    stems: tuple[str, ...]  # image stems (no extension)
    image_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a YOLO dataset (train/val/test) from RipVIS with a video-level stratified split.\n\n"
            "Rules:\n"
            "- test = RipVIS/val (images + labels)\n"
            "- train/val split from RipVIS/train cleaned_images (video-level stratified)\n"
            "- additional_data goes fully into train (only items that have matching labels)\n"
        )
    )
    p.add_argument("--ripvis-root", type=Path, default=Path("RipVIS"))
    p.add_argument("--out", type=Path, required=True, help="Output dataset root.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help=(
            "How to materialize files in the output dataset. "
            "'hardlink' is fastest and uses no extra disk space, but requires same filesystem."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Compute split and counts without writing files.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying if destination file already exists.",
    )
    p.add_argument(
        "--dedup-train-by-fps",
        action="store_true",
        help=(
            "Deduplicate train cleaned_images frames by time using hardcoded RipVIS FPS "
            f"({RIPVIS_FPS_VERSION}). Applies only to video-style frames (RipVIS-*/RipVIS-NR-*)."
        ),
    )
    p.add_argument(
        "--min-dt",
        type=float,
        default=1.0,
        help="Minimum time difference (seconds) between kept frames during dedup (train only).",
    )
    return p.parse_args()


def _iter_images(dir_path: Path) -> Iterable[Path]:
    for e in os.scandir(dir_path):
        if not e.is_file():
            continue
        p = Path(e.path)
        if p.suffix.lower() in _IMG_EXTS:
            yield p


def _link_or_copy(src: Path, dst: Path, mode: str, *, skip_existing: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and (dst.exists() or dst.is_symlink()):
        return
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            return
        os.symlink(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            return
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def _ensure_empty_label(dst: Path, *, skip_existing: bool, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dst.exists():
        return
    if dry_run:
        return
    dst.write_text("", encoding="utf-8")


def polygon_area_norm(coords: list[float]) -> float:
    n = len(coords) // 2
    if n < 3:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) / 2.0


def instance_area_from_yolo_line_norm(line: str) -> float:
    parts = line.strip().split()
    if len(parts) < 2:
        return 0.0
    coords = parts[1:]
    try:
        vals = list(map(float, coords))
    except ValueError:
        return 0.0
    if len(vals) % 2 != 0:
        return 0.0
    if len(vals) == 4:
        # bbox: x_center y_center w h (normalized)
        w = float(vals[2])
        h = float(vals[3])
        return max(w, 0.0) * max(h, 0.0)
    if len(vals) < 6:
        return 0.0
    return polygon_area_norm(vals)


def frame_score_from_label_file_norm(label_path: Path) -> float:
    if not label_path.is_file():
        return 0.0
    total = 0.0
    with label_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            total += instance_area_from_yolo_line_norm(ln)
    return total


def read_orientation(image_path: Path) -> str:
    w, h = Image.open(image_path).size
    return "H" if w >= h else "V"


_RE_POS = re.compile(r"^RipVIS-(?P<video>\d+)_\d+$")
_RE_NEG = re.compile(r"^RipVIS-NR-(?P<video>\d+)_\d+$")
_RE_POS_FRAME = re.compile(r"^RipVIS-(?P<video>\d+?)_(?P<frame>\d+)$")
_RE_NEG_FRAME = re.compile(r"^RipVIS-NR-(?P<video>\d+?)_(?P<frame>\d+)$")


def parse_video_key(stem: str) -> tuple[str, str, str]:
    m = _RE_NEG.match(stem)
    if m:
        vid = m.group("video")
        return f"N-{vid}", "negative", vid
    m = _RE_POS.match(stem)
    if m:
        vid = m.group("video")
        return f"P-{vid}", "positive", vid
    raise ValueError(f"Unrecognized stem format: {stem}")


def parse_video_id_and_frame_id(stem: str) -> tuple[str, int] | None:
    """
    Return (video_id, frame_id) for stems like:
    - RipVIS-003_00024
    - RipVIS-NR-002_00024
    """
    m = _RE_NEG_FRAME.match(stem)
    if m:
        return m.group("video").zfill(3), int(m.group("frame"))
    m = _RE_POS_FRAME.match(stem)
    if m:
        return m.group("video").zfill(3), int(m.group("frame"))
    return None


def assign_difficulties(positive_videos: list[VideoInfo]) -> dict[str, str]:
    scored = [(v.video_key, float(v.score or 0.0)) for v in positive_videos]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 0:
        return {}
    a = n // 3
    b = (2 * n) // 3
    out: dict[str, str] = {}
    for i, (k, _s) in enumerate(scored):
        if i < a:
            out[k] = "Easy"
        elif i < b:
            out[k] = "Medium"
        else:
            out[k] = "Hard"
    return out


def stratified_split_videos(
    videos: list[VideoInfo],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], dict[str, int]]:
    rng = random.Random(seed)
    strata: DefaultDict[str, list[str]] = DefaultDict(list)
    for v in videos:
        if v.bucket == "positive":
            key = f"{v.orientation}_{v.difficulty}"
        else:
            key = v.orientation
        strata[key].append(v.video_key)

    train_keys: set[str] = set()
    val_keys: set[str] = set()
    stats: dict[str, int] = {}

    for k, keys in strata.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_val = int(round(n * float(val_ratio)))
        n_val = max(0, min(n, n_val))
        val_part = keys[:n_val]
        train_part = keys[n_val:]
        train_keys.update(train_part)
        val_keys.update(val_part)
        stats[k] = n

    return train_keys, val_keys, stats


def main() -> None:
    args = parse_args()
    ripvis_root: Path = args.ripvis_root
    out_root: Path = args.out

    if not ripvis_root.is_dir():
        raise SystemExit(f"--ripvis-root not found: {ripvis_root}")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1 (exclusive).")

    cleaned_images = ripvis_root / "train" / "sampled_images" / "cleaned_images"
    cleaned_labels = ripvis_root / "train" / "yolo_annotations" / "labels"

    add_images = ripvis_root / "train" / "sampled_images" / "additional_data" / "images"
    add_labels = ripvis_root / "train" / "yolo_annotations" / "additional_data" / "labels"

    val_images = ripvis_root / "val" / "sampled_images" / "images"
    val_labels = ripvis_root / "val" / "yolo_annotations" / "labels"

    for p in (cleaned_images, cleaned_labels, add_images, add_labels, val_images, val_labels):
        if not p.exists():
            raise SystemExit(f"Required path missing: {p}")

    # Output structure
    out_train_images = out_root / "train" / "images"
    out_train_labels = out_root / "train" / "labels"
    out_val_images = out_root / "val" / "images"
    out_val_labels = out_root / "val" / "labels"
    out_test_images = out_root / "test" / "images"
    out_test_labels = out_root / "test" / "labels"

    # 1) Index cleaned_images by video
    videos_images: DefaultDict[str, list[Path]] = DefaultDict(list)
    meta_bucket: dict[str, tuple[str, str]] = {}
    for img_path in _iter_images(cleaned_images):
        stem = img_path.stem
        video_key, bucket, video_id = parse_video_key(stem)
        videos_images[video_key].append(img_path)
        meta_bucket[video_key] = (bucket, video_id)

    # 2) Build VideoInfo list (orientation + score)
    video_infos: list[VideoInfo] = []
    pos_infos: list[VideoInfo] = []

    for video_key, paths in tqdm(
        sorted(videos_images.items(), key=lambda kv: kv[0]),
        desc="Index videos (orientation + score)",
    ):
        paths = sorted(paths, key=lambda p: p.name)
        bucket, video_id = meta_bucket[video_key]
        orientation = read_orientation(paths[0])

        stems = tuple(p.stem for p in paths)
        if bucket == "positive":
            frame_scores = [frame_score_from_label_file_norm(cleaned_labels / f"{s}.txt") for s in stems]
            score = float(sum(frame_scores) / max(1, len(frame_scores)))
        else:
            score = None

        info = VideoInfo(
            video_key=video_key,
            bucket=bucket,
            video_id=video_id,
            orientation=orientation,
            difficulty=None,
            score=score,
            stems=stems,
            image_paths=tuple(paths),
        )
        video_infos.append(info)
        if bucket == "positive":
            pos_infos.append(info)

    # 3) Assign positive difficulty bins
    diff_map = assign_difficulties(pos_infos)
    video_infos = [
        (VideoInfo(**{**v.__dict__, "difficulty": diff_map.get(v.video_key)}) if v.bucket == "positive" else v)
        for v in video_infos
    ]

    # 4) Stratified split at video-level
    train_video_keys, val_video_keys, strata_sizes = stratified_split_videos(
        video_infos,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    # 5) Copy cleaned_images into train/val based on video split
    split_for_video: dict[str, str] = {}
    split_for_video.update({k: "train" for k in train_video_keys})
    split_for_video.update({k: "val" for k in val_video_keys})

    # Copy cleaned images (outermost progress only)
    cleaned_total_images = sum(len(v.image_paths) for v in video_infos)
    cleaned_copied_images = 0
    cleaned_labels_copied = 0
    cleaned_labels_empty = 0
    cleaned_skipped_by_dedup = 0
    dedup_videos_affected = 0
    dedup_videos_missing_fps = 0

    fps_map = ripvis_fps_v184() if bool(args.dedup_train_by_fps) else {}

    for v in tqdm(video_infos, desc="Copy cleaned_images videos"):
        split = split_for_video.get(v.video_key)
        if split is None:
            raise RuntimeError(f"Video {v.video_key} not assigned to a split.")

        if split == "train":
            dst_img_dir, dst_lbl_dir = out_train_images, out_train_labels
        else:
            dst_img_dir, dst_lbl_dir = out_val_images, out_val_labels

        last_kept_frame_id: int | None = None
        fps: float | None = None
        if split == "train" and bool(args.dedup_train_by_fps):
            fps = fps_map.get(v.video_id)
            if fps is None:
                dedup_videos_missing_fps += 1

        kept_in_video = 0
        skipped_in_video = 0

        for img_path in v.image_paths:
            if split == "train" and bool(args.dedup_train_by_fps) and fps is not None:
                parsed = parse_video_id_and_frame_id(img_path.stem)
                if parsed is not None:
                    _video_id, frame_id = parsed
                    if last_kept_frame_id is None:
                        last_kept_frame_id = frame_id
                    else:
                        dt = (frame_id - last_kept_frame_id) / float(fps)
                        if dt < float(args.min_dt):
                            cleaned_skipped_by_dedup += 1
                            skipped_in_video += 1
                            continue
                        last_kept_frame_id = frame_id

            dst_img = dst_img_dir / img_path.name
            if not args.dry_run:
                _link_or_copy(img_path, dst_img, args.mode, skip_existing=bool(args.skip_existing))
            cleaned_copied_images += 1
            kept_in_video += 1

            src_lbl = cleaned_labels / f"{img_path.stem}.txt"
            dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"
            if src_lbl.is_file():
                if not args.dry_run:
                    _link_or_copy(src_lbl, dst_lbl, args.mode, skip_existing=bool(args.skip_existing))
                cleaned_labels_copied += 1
            else:
                _ensure_empty_label(dst_lbl, skip_existing=bool(args.skip_existing), dry_run=bool(args.dry_run))
                cleaned_labels_empty += 1

        if split == "train" and bool(args.dedup_train_by_fps) and skipped_in_video > 0:
            dedup_videos_affected += 1

    if not bool(args.dedup_train_by_fps):
        assert cleaned_copied_images == cleaned_total_images

    # 6) Additional data -> all to train (only those with matching label)
    add_imgs_total = 0
    add_imgs_copied = 0
    add_imgs_skipped_no_label = 0

    for img_path in tqdm(sorted(_iter_images(add_images), key=lambda p: p.name), desc="Copy additional_data"):
        add_imgs_total += 1
        lbl_path = add_labels / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            add_imgs_skipped_no_label += 1
            continue

        if not args.dry_run:
            _link_or_copy(img_path, out_train_images / img_path.name, args.mode, skip_existing=bool(args.skip_existing))
            _link_or_copy(lbl_path, out_train_labels / lbl_path.name, args.mode, skip_existing=bool(args.skip_existing))
        add_imgs_copied += 1

    # 7) Test = RipVIS/val
    test_imgs_total = 0
    test_imgs_copied = 0
    test_labels_missing = 0

    for img_path in tqdm(sorted(_iter_images(val_images), key=lambda p: p.name), desc="Copy test (RipVIS val)"):
        test_imgs_total += 1
        lbl_path = val_labels / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            test_labels_missing += 1
            continue
        if not args.dry_run:
            _link_or_copy(img_path, out_test_images / img_path.name, args.mode, skip_existing=bool(args.skip_existing))
            _link_or_copy(lbl_path, out_test_labels / lbl_path.name, args.mode, skip_existing=bool(args.skip_existing))
        test_imgs_copied += 1

    def _count_images(dir_path: Path) -> int:
        if not dir_path.is_dir():
            return 0
        return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)

    def _count_labels(dir_path: Path) -> int:
        if not dir_path.is_dir():
            return 0
        return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    # Final report
    n_train_videos = sum(1 for v in video_infos if split_for_video.get(v.video_key) == "train")
    n_val_videos = sum(1 for v in video_infos if split_for_video.get(v.video_key) == "val")
    n_pos_videos = sum(1 for v in video_infos if v.bucket == "positive")
    n_neg_videos = sum(1 for v in video_infos if v.bucket == "negative")

    print("=== RipVIS YOLO dataset build report ===")
    print(f"Dry run: {bool(args.dry_run)}")
    print(f"Mode: {args.mode}")
    print(f"Output root: {out_root}")
    print("--- Video split summary (cleaned_images) ---")
    print(f"Videos total: {len(video_infos)} (positive={n_pos_videos}, negative={n_neg_videos})")
    print(f"Videos train: {n_train_videos}")
    print(f"Videos val:   {n_val_videos}")
    print("Strata sizes:")
    for k in sorted(strata_sizes.keys()):
        print(f"  {k}: {strata_sizes[k]}")
    print("--- cleaned_images copy ---")
    print(f"Images copied: {cleaned_copied_images}")
    print(f"Labels copied: {cleaned_labels_copied}")
    print(f"Empty labels created (neg/missing): {cleaned_labels_empty}")
    if bool(args.dedup_train_by_fps):
        print("--- train dedup (pre-copy) ---")
        print(f"Enabled: True (fps_table={RIPVIS_FPS_VERSION}, min_dt={float(args.min_dt)})")
        print(f"Frames skipped by dedup: {cleaned_skipped_by_dedup}")
        print(f"Videos affected: {dedup_videos_affected}")
        print(f"Videos missing FPS (dedup skipped for those videos): {dedup_videos_missing_fps}")
    print("--- additional_data copy (train only) ---")
    print(f"Images scanned: {add_imgs_total}")
    print(f"Images copied (had label): {add_imgs_copied}")
    print(f"Images skipped (no label): {add_imgs_skipped_no_label}")
    print("--- test copy (RipVIS val) ---")
    print(f"Images scanned: {test_imgs_total}")
    print(f"Images copied (had label): {test_imgs_copied}")
    print(f"Images skipped (missing label): {test_labels_missing}")
    print("--- final split counts ---")
    if bool(args.dry_run):
        print("Dry run enabled: reporting intended counts (not filesystem).")
        train_images_intended = cleaned_copied_images + add_imgs_copied
        train_labels_intended = cleaned_copied_images + add_imgs_copied  # empty labels for cleaned_images are created
        val_images_intended = cleaned_total_images - cleaned_copied_images
        val_labels_intended = val_images_intended  # empty labels for cleaned_images are created
        test_images_intended = test_imgs_copied
        test_labels_intended = test_imgs_copied

        print(f"train/images: {train_images_intended}")
        print(f"train/labels: {train_labels_intended}")
        print(f"val/images:   {val_images_intended}")
        print(f"val/labels:   {val_labels_intended}")
        print(f"test/images:  {test_images_intended}")
        print(f"test/labels:  {test_labels_intended}")
    else:
        print("Reporting filesystem counts.")
        print(f"train/images: {_count_images(out_train_images)}")
        print(f"train/labels: {_count_labels(out_train_labels)}")
        print(f"val/images:   {_count_images(out_val_images)}")
        print(f"val/labels:   {_count_labels(out_val_labels)}")
        print(f"test/images:  {_count_images(out_test_images)}")
        print(f"test/labels:  {_count_labels(out_test_labels)}")


if __name__ == "__main__":
    main()


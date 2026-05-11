"""Convert TartanDrive rosbags to the scaffold's on-disk format.

For each input `.bag`, reads two topics — RGB image and Twist command — and
emits a per-trajectory directory in the layout consumed by
`TartanDriveDataSource`:

    out_dir/
      <bag_stem>/
        frames.npy   # uint8 [T, H, W, 3]   (already resized to args.resolution)
        actions.npy  # float32 [T, 2]       (throttle = linear.x, steer = angular.z)
        meta.json    # {length, source_bag, fps, image_topic, action_topic}

Bags missing either topic are skipped with a warning.

Time alignment: images and cmds arrive at different rates (~20 Hz and ~100 Hz
in TartanDrive 1.0). We treat image timestamps as the reference grid, decimate
by `--decimate` (default 2 -> 10 Hz), and for each image take the most-recent
cmd at-or-before the image timestamp (zero-order hold). This is what the
upstream `rosbag_to_dataset` spec does at dt=0.1.

Inspect a bag without converting:
    uv run python scripts/convert_bags.py --inspect path/to/foo.bag

Convert one or more:
    DATASET_DIR=~/data/nanowm uv run python scripts/convert_bags.py \
        --bags $DATASET_DIR/raw_bags/<shard>/*.bag \
        --out_dir $DATASET_DIR/tartandrive_staging \
        --resolution 256 --decimate 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


# Default topic names from TartanDrive's `specs/2021_atv_3.yaml` spec.
DEFAULT_IMAGE_TOPIC = "/multisense/left/image_rect_color"
DEFAULT_ACTION_TOPIC = "/cmd"


def list_bag_topics(bag_path: Path) -> None:
    """Print all topics + message types + message counts in `bag_path`."""
    with AnyReader([bag_path]) as reader:
        print(f"Bag: {bag_path}")
        print(f"  duration: {(reader.end_time - reader.start_time) / 1e9:.2f} s")
        print(f"  topics ({len(reader.connections)}):")
        for c in reader.connections:
            print(f"    {c.topic:<48} {c.msgtype:<40} ({c.msgcount} msgs)")


def _decode_image(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image to a uint8 [H, W, 3] RGB array.

    Handles the encodings TartanDrive actually emits (`rgb8`, `bgr8`); other
    encodings raise so we fail loudly rather than silently miscolor.
    """
    h, w = int(msg.height), int(msg.width)
    encoding = msg.encoding
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return raw.reshape(h, w, 3)
    if encoding == "bgr8":
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()  # BGR -> RGB
    raise NotImplementedError(
        f"Unsupported image encoding: {encoding!r}. "
        "Add a branch in _decode_image() if your bag uses something else."
    )


def _resize(frame: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a uint8 [H, W, 3] to [resolution, resolution, 3] via bilinear PIL."""
    return np.asarray(
        Image.fromarray(frame).resize((resolution, resolution), Image.BILINEAR)
    )


def _zero_order_hold_actions(
    image_times_ns: np.ndarray,  # int64 [T]
    action_times_ns: np.ndarray,  # int64 [N]
    action_values: np.ndarray,  # float32 [N, 2]
) -> np.ndarray:
    """For each image timestamp, pick the most-recent action at-or-before it.

    Returns float32 [T, 2]. If an image precedes all action stamps it inherits
    the first action (rare; happens only at trajectory edges).
    """
    # np.searchsorted with side='right' gives us the insertion index, so the
    # action at index-1 is the most-recent at-or-before each image stamp.
    idx = np.searchsorted(action_times_ns, image_times_ns, side="right") - 1
    idx = np.clip(idx, 0, len(action_times_ns) - 1)
    return action_values[idx]


def convert_bag(
    bag_path: Path,
    out_dir: Path,
    image_topic: str,
    action_topic: str,
    resolution: int,
    decimate: int,
) -> Optional[dict]:
    """Convert one bag. Returns meta dict on success, None if skipped."""
    bag_stem = bag_path.stem
    out_traj_dir = out_dir / bag_stem
    out_frames = out_traj_dir / "frames.npy"
    out_actions = out_traj_dir / "actions.npy"
    out_meta = out_traj_dir / "meta.json"

    # Resumable: skip if all outputs already exist
    if out_frames.exists() and out_actions.exists() and out_meta.exists():
        print(f"  [{bag_stem}] already converted, skipping")
        with open(out_meta) as f:
            return json.load(f)

    with AnyReader([bag_path]) as reader:
        # Collect connections for the two topics
        image_conns = [c for c in reader.connections if c.topic == image_topic]
        action_conns = [c for c in reader.connections if c.topic == action_topic]
        if not image_conns:
            print(f"  [{bag_stem}] skip: no {image_topic!r}; "
                  f"available: {sorted({c.topic for c in reader.connections})[:5]}...")
            return None
        if not action_conns:
            print(f"  [{bag_stem}] skip: no {action_topic!r}")
            return None

        # Stream image messages, decimating by `decimate`. We decode lazily and
        # resize immediately to keep memory bounded.
        frames: List[np.ndarray] = []
        image_times_ns: List[int] = []
        for kept_idx, (connection, t_ns, raw) in enumerate(
            reader.messages(connections=image_conns)
        ):
            if kept_idx % decimate != 0:
                continue
            msg = reader.deserialize(raw, connection.msgtype)
            img = _decode_image(msg)
            img = _resize(img, resolution)
            frames.append(img)
            image_times_ns.append(int(t_ns))

        if not frames:
            print(f"  [{bag_stem}] skip: no image messages decoded")
            return None

        # Stream action messages (cheap; small).
        action_msgtype = action_conns[0].msgtype
        # TartanDrive 1.0's /cmd is published as TwistStamped (Twist wrapped
        # with a header). Both shapes are supported; we reach into .twist
        # when needed.
        is_twist_stamped = action_msgtype.endswith("/TwistStamped")
        is_twist = action_msgtype.endswith("/Twist")
        if not (is_twist or is_twist_stamped):
            raise NotImplementedError(
                f"Unsupported action message type: {action_msgtype}. "
                "Expected Twist or TwistStamped."
            )

        action_times_ns: List[int] = []
        action_values: List[Tuple[float, float]] = []
        for connection, t_ns, raw in reader.messages(connections=action_conns):
            msg = reader.deserialize(raw, connection.msgtype)
            twist = msg.twist if is_twist_stamped else msg
            # linear.x = throttle, angular.z = steering
            action_times_ns.append(int(t_ns))
            action_values.append((float(twist.linear.x), float(twist.angular.z)))

        if not action_values:
            print(f"  [{bag_stem}] skip: no action messages decoded")
            return None

    frames_np = np.stack(frames)  # uint8 [T, R, R, 3]
    image_ts = np.asarray(image_times_ns, dtype=np.int64)
    action_ts = np.asarray(action_times_ns, dtype=np.int64)
    action_vals = np.asarray(action_values, dtype=np.float32)

    actions_np = _zero_order_hold_actions(image_ts, action_ts, action_vals)

    T = len(frames_np)
    if T < 16:
        print(f"  [{bag_stem}] skip: only {T} frames after decimation (< 16)")
        return None

    duration_s = float((image_ts[-1] - image_ts[0]) / 1e9)
    fps = float((T - 1) / duration_s) if duration_s > 0 else 0.0

    out_traj_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_frames, frames_np)
    np.save(out_actions, actions_np)
    meta = {
        "length": int(T),
        "source_bag": str(bag_path),
        "source_traj_id": bag_stem,
        "fps": fps,
        "image_topic": image_topic,
        "action_topic": action_topic,
        "decimate": int(decimate),
        "resolution": int(resolution),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{bag_stem}] T={T} fps={fps:.2f} action_range=[{action_vals.min():.2f}, {action_vals.max():.2f}]")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inspect", type=Path, default=None,
        help="Print topic listing for ONE bag and exit (no conversion).",
    )
    parser.add_argument(
        "--bags", nargs="+", type=Path, default=[],
        help="One or more .bag paths to convert.",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        help="Output staging dir; per-bag subdirs are created here.",
    )
    parser.add_argument("--image_topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--action_topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--decimate", type=int, default=2,
        help="Keep 1 of every N image messages (default 2 = 20 Hz -> 10 Hz).",
    )
    args = parser.parse_args()

    if args.inspect is not None:
        list_bag_topics(args.inspect)
        return 0

    if not args.bags:
        parser.error("--bags is required unless --inspect is given")
    if args.out_dir is None:
        parser.error("--out_dir is required for conversion")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(args.bags)} bag(s) -> {args.out_dir}")
    print(f"  image_topic={args.image_topic}  action_topic={args.action_topic}")
    print(f"  resolution={args.resolution}  decimate={args.decimate}")

    n_ok = 0
    for i, bag in enumerate(args.bags, 1):
        print(f"\n[{i}/{len(args.bags)}] {bag.name}")
        if not bag.exists():
            print(f"  skip: not found")
            continue
        try:
            meta = convert_bag(
                bag, args.out_dir,
                image_topic=args.image_topic,
                action_topic=args.action_topic,
                resolution=args.resolution,
                decimate=args.decimate,
            )
            if meta is not None:
                n_ok += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue

    print(f"\nDone. Converted {n_ok}/{len(args.bags)} bag(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

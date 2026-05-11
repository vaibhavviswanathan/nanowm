# TartanDrive data acquisition — reality check

The scaffold README said "~30 GB for raw TartanDrive 1.0". The actual bucket
contents tell a different story.

## What's actually published

Endpoint: `https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive`
Auth: anonymous (UNSIGNED), Swift-compatible (use direct HTTP, not boto3 — its
AWS region-redirect logic breaks on Swift).

Contents (listed via `?format=json`):
- 23 tarballs named `YYYYMMDD_heightmaps_N.tar.gz`
- Smallest: **69 GB** (`20210828_heightmaps_1.tar.gz`)
- Largest: **127 GB** (`20210910_heightmaps_1.tar.gz`)
- Total: **~2 TB compressed**

The "heightmaps" in the names is misleading — based on the rosbag_to_dataset
spec (`specs/2021_atv_3.yaml` in https://github.com/castacks/tartan_drive),
each tarball is per-session rosbags containing all modalities: RGB images,
heightmaps, IMU, wheel RPM, shock travel, odometry, cmd (action).

## To use it we need ROS

The published format is rosbags. Conversion to per-trajectory torch tensors
requires:
- ROS melodic (Python 2-era; or noetic / pure-Python `rosbag` reader)
- The repo's `rosbag_to_dataset` package
- Custom message types (`physics_atv_racepak`)
- An extracted-trajectory output dir with keys like `image_rgb`, `cmd`, etc.

## TartanDrive 2.0

Per `https://github.com/castacks/tartan_drive_2.0`: supports a KITTI-format
download via a GUI tool. Status "Under construction"; documentation is
incomplete. Worth trying if we want preprocessed data, but no guarantees.

## Free disk

Local: 431 GB free, 16 GB GPU.

## Options

A. **Synthetic substitute (recommended for the engineering POC)**: write a
   tiny generator that produces frames + actions in our scaffold's on-disk
   layout. Train end-to-end on ~30 minutes of synthetic data (the engineering
   plan stays exactly the same; only the loader's input changes). After the
   pipeline is green and tagged, swap in real TartanDrive as a separate work
   stream. Time cost: ~30 min for the generator.

B. **Download one TartanDrive 1.0 shard (~69 GB)** and stand up ROS / rosbag
   extraction. Time cost: ~1 day to install ROS, extract a shard, and verify
   the data flows through preprocess.

C. **Try TartanDrive 2.0 KITTI export**. Time cost: unknown — needs the GUI
   tool which we have to run interactively. Format/size opaque.

D. **Use a smaller off-road dataset** (RUGD ~40 GB, ORFD ~30 GB) and remap
   actions if possible. Time cost: ~half day to integrate.

#!/usr/bin/env python3
"""
stabilize_realsense.py

Stabilize a RealSense D455 stream recorded in a rosbag using raw Gyroscope data.
This script correctly integrates high-rate angular velocities, interpolates them
to exact image frame timestamps, and uses a heavy Gaussian low-pass filter to 
simulate a perfectly locked "tripod" hover state.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def resolve_typestore(name):
    try:
        store_enum = getattr(Stores, name)
    except AttributeError:
        valid = [s.name for s in Stores]
        raise SystemExit(
            f"Unknown --typestore '{name}'. Valid options:\n  " + "\n  ".join(valid)
        )
    return get_typestore(store_enum)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def decode_image_msg(msg, msgtype):
    if "CompressedImage" in msgtype:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8",):
        return buf.reshape(h, w, 3)
    elif enc in ("rgb8",):
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif enc in ("mono8",):
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    else:
        raise RuntimeError(f"Unsupported encoding '{msg.encoding}'.")


def read_bag_data(bag_path, image_topic, imu_topic, typestore):
    """Reads image timestamps and raw gyro data from the bag."""
    frames = []
    imus = []
    
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in (image_topic, imu_topic)]
        if not conns:
            raise RuntimeError("Required topics not found in the bag.")
            
        for connection, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = stamp_to_sec(msg.header.stamp)
            
            if connection.topic == image_topic:
                frames.append((t, connection.msgtype, rawdata))
            elif connection.topic == imu_topic:
                # Extract raw angular velocity (rad/s) from RealSense IMU
                gx = msg.angular_velocity.x
                gy = msg.angular_velocity.y
                gz = msg.angular_velocity.z
                imus.append((t, gx, gy, gz))
                
    frames.sort(key=lambda x: x[0])
    imus.sort(key=lambda x: x[0])
    return frames, imus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", default='~/bags/rosbag_1')
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--imu-topic", default="/camera/camera/imu")
    ap.add_argument("--out", default="stabilized_smooth2.mp4")
    ap.add_argument("--crop-frac", type=float, default=1)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--typestore", default="ROS2_HUMBLE")
    ap.add_argument("--smooth-window", type=float, default=30.0, 
                    help="Gaussian filter sigma (in frames). Higher = more locked-in stabilization.")

    # Camera Intrinsics
    ap.add_argument("--fx", type=float, default=393.12779563)
    ap.add_argument("--fy", type=float, default=394.76440916)
    ap.add_argument("--cx", type=float, default=321.48263787)
    ap.add_argument("--cy", type=float, default=241.58044155)
    
    args = ap.parse_args()

    typestore = resolve_typestore(args.typestore)
    
    print("Indexing bag data...")
    frames, imus = read_bag_data(args.bag, args.image_topic, args.imu_topic, typestore)
    
    if len(frames) < 2 or len(imus) < 2:
        print("Insufficient frames or IMU data.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(frames)} frames, {len(imus)} IMU messages")
    
    frame_times = np.array([f[0] for f in frames])
    fps = args.fps if args.fps else (1.0 / np.median(np.diff(frame_times)))

    # ---- 1. Integrate Gyroscope Data Continuously ----
    print("Integrating RealSense IMU data...")
    imu_times = np.array([i[0] for i in imus])
    gyros = np.array([[i[1], i[2], i[3]] for i in imus])
    
    # Calculate dt between IMU messages
    dt = np.diff(imu_times)
    dt = np.insert(dt, 0, 0.0)
    
    # Continuous Euler integration of angular velocity to get orientation (radians)
    continuous_angles = np.cumsum(gyros * dt[:, np.newaxis], axis=0)

    # ---- 2. Interpolate to Frame Timestamps ----
    print("Interpolating and smoothing trajectory...")
    # Create interpolation function based on continuous IMU timeline
    interp_func = interp1d(imu_times, continuous_angles, axis=0, kind='linear', fill_value="extrapolate")
    
    # Extract exact camera angles at the precise microsecond of frame exposure
    frame_angles = interp_func(frame_times)

    # ---- 3. Heavy Gaussian Smoothing (Hover Lock) ----
    # This replaces the dead_zone filter. It creates a perfectly smooth desired path.
    # Because gyro integration naturally drifts, this filter also automatically subtracts the drift!
    smoothed_angles = gaussian_filter1d(frame_angles, sigma=args.smooth_window, axis=0)
    
    # The jitter we want to cancel is the difference between actual orientation and the smoothed path
    angle_residuals = frame_angles - smoothed_angles

    # ---- 4. Prepare Camera Intrinsics ----
    K = np.array([
        [args.fx, 0,       args.cx],
        [0,       args.fy, args.cy],
        [0,       0,       1      ]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    h, w = None, None
    crop_w, crop_h, x0, y0 = 0, 0, 0, 0
    writer = None

    # ---- 5. Warp and Write ----
    print("Applying smooth correction and writing output...")
    with AnyReader([Path(args.bag)], default_typestore=typestore) as reader:
        for idx, (t, msgtype, rawdata) in enumerate(frames):
            msg = reader.deserialize(rawdata, msgtype)
            img = decode_image_msg(msg, msgtype)
            
            if writer is None:
                h, w = img.shape[:2]
                crop_w = int(w * args.crop_frac)
                crop_h = int(h * args.crop_frac)
                x0 = (w - crop_w) // 2
                y0 = (h - crop_h) // 2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, fps, (crop_w, crop_h))

            # Realsense specific axis mapping might be required here if roll/pitch/yaw are swapped
            # e.g., if X is pitch instead of roll. We apply the inverse residual to cancel motion.
            jitter_correction_euler = -angle_residuals[idx]
            R_corr = R.from_euler('xyz', jitter_correction_euler, degrees=False).as_matrix()

            # H = K * R * K^-1
            H = K @ R_corr @ K_inv

            warped = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cropped = warped[y0:y0 + crop_h, x0:x0 + crop_w]
            writer.write(cropped)

            if (idx + 1) % 200 == 0:
                print(f"  {idx + 1}/{len(frames)}")

    if writer:
        writer.release()
    print(f"Done. Wrote {args.out} at {crop_w}x{crop_h}")


if __name__ == "__main__":
    main()
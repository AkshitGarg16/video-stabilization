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


def read_bag_data(bag, image_topic, odom_topic, imu_topic, typestore):
    """Reads image timestamps, odometry (absolute position + orientation), and
    IMU angular velocity from the bag."""
    frames = []
    odometry = []
    imu = []

    with AnyReader([Path(bag)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in (image_topic, odom_topic, imu_topic)]
        if not conns:
            raise RuntimeError("Required topics not found in the bag.")

        for connection, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = stamp_to_sec(msg.header.stamp)

            if connection.topic == image_topic:
                frames.append((t, connection.msgtype, rawdata))
            elif connection.topic == odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odometry.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif connection.topic == imu_topic:
                w = msg.angular_velocity
                imu.append((t, w.x, w.y, w.z))

    frames.sort(key=lambda x: x[0])
    odometry.sort(key=lambda x: x[0])
    imu.sort(key=lambda x: x[0])
    return frames, odometry, imu


def gyro_propagate_orientation(odom_times, odom_quats, imu_times, imu_omega, frame_times, substeps=5):
    """Interpolates orientation at frame_times by integrating IMU gyro
    (angular velocity) forward from the bracketing odom sample, then
    distributing the drift correction needed to land exactly on the next
    odom sample. This respects the vehicle's actual rotational path between
    sparse odom updates, unlike linearly interpolating Euler angles (which
    assumes the pose moves in a straight line through angle-space and can
    cut corners during fast or combined-axis rotation)."""
    omega_interp = interp1d(imu_times, imu_omega, axis=0, kind="linear", fill_value="extrapolate")
    odom_rot = R.from_quat(odom_quats)

    n = len(odom_times)
    idx = np.clip(np.searchsorted(odom_times, frame_times, side="right") - 1, 0, n - 2)

    def integrate(t0, t_end):
        ts = np.linspace(t0, t_end, substeps + 1)
        rotvec = np.trapezoid(omega_interp(ts), ts, axis=0)
        return R.from_rotvec(rotvec)

    out_quats = np.empty((len(frame_times), 4))
    for k, t in enumerate(frame_times):
        i = idx[k]
        t0, t1 = odom_times[i], odom_times[i + 1]
        q0, q1 = odom_rot[i], odom_rot[i + 1]
        span = t1 - t0
        if span <= 0:
            out_quats[k] = q0.as_quat()
            continue

        t_clamped = np.clip(t, t0, t1)
        frac = (t_clamped - t0) / span

        prop_partial = integrate(t0, t_clamped)
        prop_full = integrate(t0, t1)
        drift = prop_full.inv() * q0.inv() * q1
        partial_drift = R.from_rotvec(drift.as_rotvec() * frac)

        out_quats[k] = (q0 * prop_partial * partial_drift).as_quat()

    return R.from_quat(out_quats)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=str, default="/home/akshit/bags/2026-07-26_11-30-11", help="Path to the ROS bag file")
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--odom-topic", default="/mavros/local_position/odom")
    ap.add_argument("--imu-topic", default="/mavros/imu/data",
                     help="IMU topic used to gyro-propagate orientation between odom samples.")
    ap.add_argument("--out", default="stabilized_odom3.mp4")
    ap.add_argument("--crop-frac", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--typestore", default="ROS2_HUMBLE")

    # Gaussian filter sigmas define the intended hover path (the path we
    # correct *towards*). Specified in seconds so tuning is independent of
    # frame rate, and split per-axis since orientation jitter is faster and
    # needs a shorter time constant than translational drift to still look
    # stabilized without smearing intentional yaw/pitch/roll maneuvers.
    ap.add_argument("--smooth-sigma-pos", type=float, default=0.5,
                     help="Gaussian filter sigma for position, in seconds.")
    ap.add_argument("--smooth-sigma-angle", type=float, default=0.25,
                     help="Gaussian filter sigma for orientation, in seconds.")

    # Motion gate: correction fades from full strength at zero residual down
    # to zero once the drone's actual position/orientation has strayed this
    # far from its own smoothed (intended) path. Small residual = jitter
    # around a hover attempt; large, sustained residual = real translation/
    # maneuvering that the smoothing hasn't absorbed yet.
    ap.add_argument("--dist-thresh", type=float, default=0.35,
                     help="Position residual (m) above which correction is fully disabled.")
    ap.add_argument("--angle-thresh", type=float, default=0.6,
                     help="Orientation residual (rad) above which correction is fully disabled.")

    # Camera Intrinsics
    ap.add_argument("--fx", type=float, default=393.12779563)
    ap.add_argument("--fy", type=float, default=394.76440916)
    ap.add_argument("--cx", type=float, default=321.48263787)
    ap.add_argument("--cy", type=float, default=241.58044155)

    args = ap.parse_args()

    typestore = resolve_typestore(args.typestore)

    print("Indexing bag data...")
    frames, odometry, imu = read_bag_data(args.bag, args.image_topic, args.odom_topic, args.imu_topic, typestore)

    if len(frames) < 2 or len(odometry) < 2:
        print("Insufficient frames or odometry data.", file=sys.stderr)
        sys.exit(1)
    if len(imu) < 2:
        print("Insufficient IMU data.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(frames)} frames, {len(odometry)} odometry messages, {len(imu)} IMU messages")

    frame_times = np.array([f[0] for f in frames])
    fps = args.fps if args.fps else (1.0 / np.median(np.diff(frame_times)))

    # ---- 1. Convert odometry position/orientation (absolute, no integration needed) ----
    print("Converting odometry position...")
    odom_times = np.array([o[0] for o in odometry])
    positions = np.array([[o[1], o[2], o[3]] for o in odometry])
    quats = np.array([[o[4], o[5], o[6], o[7]] for o in odometry])  # x, y, z, w

    imu_times = np.array([m[0] for m in imu])
    imu_omega = np.array([[m[1], m[2], m[3]] for m in imu])  # rad/s, body frame

    # ---- 2. Interpolate position/orientation to frame timestamps ----
    print("Interpolating position to frame timestamps...")
    pos_interp = interp1d(odom_times, positions, axis=0, kind="linear", fill_value="extrapolate")
    frame_positions = pos_interp(frame_times)

    print("Gyro-propagating orientation to frame timestamps...")
    frame_rotations = gyro_propagate_orientation(odom_times, quats, imu_times, imu_omega, frame_times)
    frame_angles = np.unwrap(frame_rotations.as_euler("xyz", degrees=False), axis=0)

    # ---- 3. Smooth path (what the drone is "trying" to hold/do) ----
    sigma_pos_frames = max(args.smooth_sigma_pos * fps, 1e-3)
    sigma_angle_frames = max(args.smooth_sigma_angle * fps, 1e-3)
    smoothed_positions = gaussian_filter1d(frame_positions, sigma=sigma_pos_frames, axis=0)
    smoothed_angles = gaussian_filter1d(frame_angles, sigma=sigma_angle_frames, axis=0)
    position_residuals = frame_positions - smoothed_positions
    angle_residuals = frame_angles - smoothed_angles

    # ---- 4. Motion gate: how much of the residual to cancel per frame ----
    # 1.0 = fully cancel jitter (actual pose is right where the smoothed path
    # says it should be), fading to 0.0 = pass the frame through untouched
    # (actual pose has strayed far from its own smoothed path - real motion,
    # not jitter). Gating on residual *distance* rather than velocity means
    # fast-but-small jitter still gets corrected, while slow-but-sustained
    # drift away from the path still gets left alone.
    dist_alpha = np.clip(1.0 - np.linalg.norm(position_residuals, axis=1) / args.dist_thresh, 0.0, 1.0)
    angle_alpha = np.clip(1.0 - np.linalg.norm(angle_residuals, axis=1) / args.angle_thresh, 0.0, 1.0)
    alpha = np.minimum(dist_alpha, angle_alpha)

    print(f"  Correction active (alpha > 0.5) on {np.mean(alpha > 0.5) * 100:.1f}% of frames")

    # ---- 5. Prepare Camera Intrinsics ----
    K = np.array([
        [args.fx, 0,       args.cx],
        [0,       args.fy, args.cy],
        [0,       0,       1      ]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    h, w = None, None
    crop_w, crop_h, x0, y0 = 0, 0, 0, 0
    writer = None

    # ---- 6. Warp and Write ----
    print("Applying gated correction and writing output...")
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

            jitter_correction_euler = -angle_residuals[idx] * alpha[idx]
            R_corr = R.from_euler("xyz", jitter_correction_euler, degrees=False).as_matrix()

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

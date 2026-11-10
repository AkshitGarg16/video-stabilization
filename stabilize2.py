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


def read_bag_data(bag_path, image_topic, odom_topic, typestore):
    """Reads image timestamps and raw gyro data from the bag."""
    frames = []
    odometry = []
    
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in (image_topic, odom_topic)]
        if not conns:
            raise RuntimeError("Required topics not found in the bag.")
            
        for connection, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = stamp_to_sec(msg.header.stamp)
            
            if connection.topic == image_topic:
                frames.append((t, connection.msgtype, rawdata))
            elif connection.topic == odom_topic:
                # Extract odom data (linear and angular velocities) from the message
                yaw = msg.twist.twist.angular.z  # Angular velocity around Z-axis (yaw rate)
                roll = msg.twist.twist.angular.x  # Angular velocity around X-axis (roll rate)
                pitch = msg.twist.twist.angular.y  # Angular velocity around Y-axis (pitch rate)
                vx = msg.twist.twist.linear.x  # Linear velocity in X direction
                vy = msg.twist.twist.linear.y  # Linear velocity in Y direction
                odometry.append((t, roll, pitch, yaw, vx, vy))

    frames.sort(key=lambda x: x[0])
    odometry.sort(key=lambda x: x[0])
    return frames, odometry


def main():
    parser = argparse.ArgumentParser(description="Video Stabilization from ROS Bag")
    parser.add_argument("bag_path", type=str, default='~/bags/rosbag_1', help="Path to the ROS bag file")
    parser.add_argument("--image-topic", type=str, default='/camera/camera/color/image_raw', help="Image topic name")
    parser.add_argument("--odom-topic", type=str, default='mavros/local_position/odom', help="Odometry topic name")
    parser.add_argument("--typestore", type=str, default="ros2", help="Typestore to use (default: ros2)")
    
    # Camera Intrinsics
    parser.add_argument("--fx", type=float, default=393.12779563)
    parser.add_argument("--fy", type=float, default=394.76440916)
    parser.add_argument("--cx", type=float, default=321.48263787)
    parser.add_argument("--cy", type=float, default=241.58044155)
    
    args = parser.parse_args()
    
    typestore = resolve_typestore(args.typestore)
    
    frames, odometry = read_bag_data(args.bag_path, args.image_topic, args.odom_topic, typestore)
    
    print(f"{len(frames)} frames, {len(odometry)} odometry messages")

    
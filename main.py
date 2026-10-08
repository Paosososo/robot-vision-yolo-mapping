"""Rotate the robot, estimate odometry, detect objects and save a map.

Edit SETTINGS below for your scene. Run: python main.py --weights /path/to/best.pt
Add --show --output-dir output/live_demo for a live annotated camera window.
All runtime functions are in this file; training and evaluation are separate.
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PREVIEW_WINDOW = 'Lab04 - live YOLO detection (Q or Esc to stop)'

# Scene paths, robot measurements and detector settings. Dimensions are placeholders.
SETTINGS = {
    "host": "localhost",
    "port": 23000,
    "scene_path": str(Path(__file__).resolve().parent / "scenes" / "Demo_CV.ttt"),  # file on the machine running CoppeliaSim
    "robot_path": "/body",
    "left_motor_path": "/body/J1",
    "right_motor_path": "/body/J2",
    "camera_path": "/body/visionSensor",
    "wheel_radius_m": 0.05,
    "wheel_track_m": 0.3,  # effective (calibrated) track; the joints are only 0.25 apart
    "left_sign": 1,
    "right_sign": 1,
    "rotation_deg": 360,
    "angular_speed_rad_s": 0.5,
    "rotation_tolerance_deg": 1,
    "max_sim_time_s": 90,
    "detect_every_steps": 5,
    "confidence": 0.5,
    "merge_radius_m": 0.35,
    "min_depth_m": 0.05,
    "max_depth_m": 8,
    "model": "yolo11n.pt",
    "clahe": False,
    "output_dir": "output/demo",
    "save_frames": True
}

def angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


@dataclass
class Odometry:
    radius: float
    track: float
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # Unwrapped, so a full revolution remains measurable.
    previous: tuple | None = None

    def update(self, left, right):
        if self.previous is not None:
            dl = self.radius * angle_delta(left, self.previous[0])
            dr = self.radius * angle_delta(right, self.previous[1])
            distance, turn = (dl + dr) / 2, (dr - dl) / self.track
            # Exact differential-drive integration, including curved motion.
            scale = math.sin(turn / 2) / (turn / 2) if abs(turn) > 1e-9 else 1.0
            self.x += distance * scale * math.cos(self.yaw + turn / 2)
            self.y += distance * scale * math.sin(self.yaw + turn / 2)
            self.yaw += turn
        self.previous = (left, right)


def project_pixel(u, v, z, width, height, fov, camera_matrix, pose):
    """Top-down RGB pixel + axial depth -> odometry map XY.

    CoppeliaSim sensor frame: +X left, +Y up, +Z forward.
    camera_matrix is the row-major 3x4 sensor-to-body transform.
    Body frame must have +X forward, +Y left, +Z up.
    """
    focal = max(width, height) / (2 * math.tan(fov / 2))
    p = [-(u - (width - 1) / 2) * z / focal,
         -(v - (height - 1) / 2) * z / focal, z]
    body = [sum(camera_matrix[4 * i + j] * p[j] for j in range(3))
            + camera_matrix[4 * i + 3] for i in range(3)]
    x, y, yaw = pose
    return (x + math.cos(yaw) * body[0] - math.sin(yaw) * body[1],
            y + math.sin(yaw) * body[0] + math.cos(yaw) * body[1])


class LandmarkMap:
    def __init__(self, merge_radius=0.35):
        self.merge_radius = merge_radius
        self.objects = []

    def observe(self, label, x, y, confidence):
        candidates = [o for o in self.objects if o['type'] == label
                      and math.hypot(o['x'] - x, o['y'] - y) < self.merge_radius]
        if candidates:
            obj = min(candidates, key=lambda o: math.hypot(o['x'] - x, o['y'] - y))
            n = obj['observations']
            obj['x'] = (obj['x'] * n + x) / (n + 1)
            obj['y'] = (obj['y'] * n + y) / (n + 1)
            obj['observations'] += 1
            obj['confidence'] = max(obj['confidence'], confidence)
        else:
            self.objects.append(dict(id=len(self.objects) + 1, type=label, x=x, y=y,
                                     confidence=confidence, observations=1))


def connect_to_robot(cfg, use_camera):
    """Connect over ZMQ, find joints and read fixed camera calibration."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient(host=cfg['host'], port=cfg['port'])
    sim = client.require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop the simulation before starting this controller.')
    # Load the scene before resolving object handles. The path is on the simulator host.
    if cfg.get('scene_path'):
        sim.loadScene(cfg['scene_path'])
    body = sim.getObject(cfg['robot_path'])
    left, right = [sim.getObject(cfg[k]) for k in ('left_motor_path', 'right_motor_path')]
    camera = sim.getObject(cfg['camera_path']) if use_camera else None
    fov = far = extrinsic = explicit = None
    if camera is not None:
        if not sim.getObjectInt32Param(camera, sim.visionintparam_perspective_operation):
            raise ValueError('Use a perspective vision sensor.')
        if sim.getObjectInt32Param(camera, sim.visionintparam_rgbignored) or sim.getObjectInt32Param(camera, sim.visionintparam_depthignored):
            raise ValueError('Enable both RGB and depth in the vision sensor settings.')
        fov = sim.getObjectFloatParam(camera, sim.visionfloatparam_perspective_angle)
        far = sim.getObjectFloatParam(camera, sim.visionfloatparam_far_clipping)
        extrinsic = sim.getObjectMatrix(camera, body)
        explicit = sim.getExplicitHandling(camera)
    return client, sim, left, right, camera, fov, far, extrinsic, explicit


def detect_and_map(sim, camera, explicit, fov, far, extrinsic, model, cfg, odom, landmarks, output, step):
    """Read aligned RGB/depth, detect objects, then update landmark locations."""
    import numpy as np
    import cv2
    if explicit:
        sim.handleVisionSensor(camera)
    raw, resolution = sim.getVisionSensorImg(camera)
    w, h = resolution
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    raw_depth, depth_resolution = sim.getVisionSensorDepth(camera, 1)
    if list(depth_resolution) != list(resolution):
        raise RuntimeError('RGB and depth resolutions differ')
    depth = np.flipud(np.frombuffer(raw_depth, dtype='<f4').reshape(h, w))
    if cfg['save_frames']:
        cv2.imwrite(str(output / f'rgb_{step:06d}.png'), bgr)
    if cfg['clahe']:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    result = model.predict(bgr, conf=cfg['confidence'], verbose=False)[0]
    for box in result.boxes:
        label = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        print(f'[Detection step {step}] {label}: confidence={confidence:.2f}')
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        # Small central patch limits contamination from box background.
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = int(round(u)), int(round(v))
        patch = depth[max(0, iy-2):min(h, iy+3), max(0, ix-2):min(w, ix+3)]
        valid = patch[np.isfinite(patch) & (patch > cfg['min_depth_m'])
                      & (patch < min(cfg['max_depth_m'], far * 0.999))]
        if valid.size == 0:
            print('  Skipped mapping: no valid depth at the bounding-box centre.')
            continue
        xy = project_pixel(u, v, float(np.median(valid)), w, h, fov,
                           extrinsic, (odom.x, odom.y, odom.yaw))
        landmarks.observe(label, *xy, confidence)
        print(f'  Map location: x={xy[0]:.2f} m, y={xy[1]:.2f} m')
    if cfg['save_frames'] or cfg.get('show', False):
        annotated = result.plot()
        if cfg['save_frames']:
            cv2.imwrite(str(output / f'frame_{step:06d}.png'), annotated)
        if cfg.get('show', False):
            cv2.imshow(PREVIEW_WINDOW, annotated)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                return False
            if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
    return True


def save_map(output, landmarks, trajectory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (output / 'objects.json').write_text(json.dumps(landmarks.objects, indent=2))
    with (output / 'odometry.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'x_m', 'y_m', 'yaw_rad'])
        writer.writerows(trajectory)
    fig, ax = plt.subplots()
    if trajectory:
        ax.plot([p[1] for p in trajectory], [p[2] for p in trajectory], label='Encoder odometry')
        x, y, yaw = trajectory[-1][1:]
        ax.arrow(x, y, 0.2 * math.cos(yaw), 0.2 * math.sin(yaw), width=0.015)
    for o in landmarks.objects:
        ax.scatter(o['x'], o['y'])
        ax.annotate(f"{o['type']} #{o['id']}", (o['x'], o['y']))
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Objects in initial robot frame', aspect='equal')
    ax.grid(True)
    fig.savefig(output / 'map.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', help='YOLO weights, e.g. runs/objects/weights/best.pt')
    parser.add_argument('--odom-only', action='store_true')
    parser.add_argument('--show', action='store_true', help='Display live annotated camera images; Q/Esc stops the scan')
    parser.add_argument('--output-dir', type=Path, help='Override SETTINGS output_dir for this run')
    args = parser.parse_args()
    if args.show and args.odom_only:
        parser.error('--show requires object detection; omit --odom-only')
    cfg = SETTINGS.copy()
    cfg['show'] = args.show
    if args.weights:
        cfg['model'] = args.weights
    if args.output_dir is not None:
        cfg['output_dir'] = str(args.output_dir)
    for key in ('wheel_radius_m', 'wheel_track_m', 'angular_speed_rad_s', 'max_sim_time_s'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['left_sign'] not in (-1, 1) or cfg['right_sign'] not in (-1, 1):
        raise ValueError('Wheel signs must be +1 or -1')
    if cfg['detect_every_steps'] < 1 or not math.isfinite(cfg['rotation_deg']):
        raise ValueError('Invalid scan configuration')
    model = None
    if not args.odom_only:
        from ultralytics import YOLO
        model = YOLO(cfg['model'])
        print(f'Loaded YOLO weights: {cfg["model"]}')
        print(f'Detector classes: {model.names}')
    output = Path(cfg['output_dir'])
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    output.mkdir(parents=True, exist_ok=True)
    client, sim, left, right, camera, fov, far, extrinsic, explicit = connect_to_robot(cfg, model is not None)
    odom = Odometry(cfg['wheel_radius_m'], cfg['wheel_track_m'])
    landmarks, trajectory = LandmarkMap(cfg['merge_radius_m']), []
    target = math.radians(cfg['rotation_deg'])
    tolerance = math.radians(cfg['rotation_tolerance_deg'])
    if tolerance <= 0:
        raise ValueError('Rotation tolerance must be positive')
    sim.setStepping(True)
    started = False
    try:
        if args.show:
            import cv2
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW, 640, 640)
            print('Live preview enabled. Click its window and press Q/Esc, or close it, to stop early.')
        sim.startSimulation()
        started = True
        sim.setJointTargetVelocity(left, 0)
        sim.setJointTargetVelocity(right, 0)
        sim.step()
        start = sim.getSimulationTime()
        step = 0
        while True:
            now = sim.getSimulationTime() - start
            # 1. Estimate robot pose from measured wheel positions.
            odom.update(cfg['left_sign'] * sim.getJointPosition(left),
                        cfg['right_sign'] * sim.getJointPosition(right))
            trajectory.append([now, odom.x, odom.y, odom.yaw])
            # 2. Detect and locate objects using the current odometry pose.
            if model and step % cfg['detect_every_steps'] == 0:
                keep_running = detect_and_map(sim, camera, explicit, fov, far, extrinsic,
                                              model, cfg, odom, landmarks, output, step)
                if not keep_running:
                    print('Scan stopped from live preview; saving the partial map.')
                    break
            # 3. Slow near the target heading and command opposite wheel speeds.
            error = target - odom.yaw
            if abs(error) <= tolerance:
                print(f'Rotation complete: {math.degrees(odom.yaw):.2f} degrees')
                break
            if now >= cfg['max_sim_time_s']:
                raise RuntimeError('Scan timed out; check wheel dimensions, joint signs and motor settings.')
            omega = max(-cfg['angular_speed_rad_s'], min(cfg['angular_speed_rad_s'], 1.5 * error))
            wheel = omega * cfg['wheel_track_m'] / (2 * cfg['wheel_radius_m'])
            sim.setJointTargetVelocity(left, -wheel * cfg['left_sign'])
            sim.setJointTargetVelocity(right, wheel * cfg['right_sign'])
            # 4. Advance CoppeliaSim one simulation step.
            sim.step()
            step += 1
    finally:
        try:
            if started:
                try:
                    sim.setJointTargetVelocity(left, 0)
                    sim.setJointTargetVelocity(right, 0)
                finally:
                    sim.stopSimulation()
        finally:
            try:
                sim.setStepping(False)
            finally:
                try:
                    save_map(output, landmarks, trajectory)
                finally:
                    if args.show:
                        import cv2
                        cv2.destroyAllWindows()
    print(f'Saved {len(landmarks.objects)} objects and map to {output}')


if __name__ == '__main__':
    main()

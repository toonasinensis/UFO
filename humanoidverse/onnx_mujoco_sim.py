"""Minimal native-MuJoCo runner for an exported UFO FB policy.

This entry point intentionally does not construct the training environment and
does not load a PyTorch checkpoint.  It only needs:

* the robot MJCF;
* a RobotState NPZ motion;
* ``FBcprAuxModel.onnx``;
* the precomputed tracking latent ``tracking_inference/zs_0.npy``.

Example:

    python -m humanoidverse.onnx_mujoco_sim \
      --model-folder runs/新数据addlelay_onnx \
      --motion humanoidverse/data/roban/named_roban_lafan_10s/aiming1_subject1_0000.npz
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/新数据addlelay_onnx"
DEFAULT_MOTION = ROOT / "humanoidverse/data/roban/named_roban_lafan_10s/aiming1_subject1_0000.npz"
DEFAULT_SCENE = ROOT / "humanoidverse/data/robots/roban_s22_handball/roban_s22_handball/xml/scene.xml"

# Deployment-only arm PD overrides. These do not modify the training YAML or
# the exported policy. Right-arm damping remains whatever was exported.
 


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_metadata_path(override: Path | None, metadata_value: str, label: str) -> Path:
    """Resolve export paths after moving a run between server and workstation."""
    if override is not None:
        path = override.expanduser().resolve()
    else:
        path = Path(metadata_value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.exists():
            parts = Path(metadata_value).parts
            if "UFO" in parts:
                suffix = parts[parts.index("UFO") + 1 :]
                local_path = ROOT.joinpath(*suffix).resolve()
                if local_path.exists():
                    path = local_path
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path

 
def _quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Quaternion multiply for xyzw quaternions."""
    x0, y0, z0, w0 = np.moveaxis(q0, -1, 0)
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    return np.stack(
        (
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ),
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by xyzw quaternions."""
    q_xyz = q[..., :3]
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (q[..., 3:4] * uv + uuv)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_inv = q.copy()
    q_inv[..., :3] *= -1.0
    return _quat_rotate(q_inv, v)


def _heading_inverse(q: np.ndarray) -> np.ndarray:
    forward = _quat_rotate(q, np.broadcast_to(np.array([1.0, 0.0, 0.0]), q.shape[:-1] + (3,)))
    half = -0.5 * np.arctan2(forward[..., 1], forward[..., 0])
    zeros = np.zeros_like(half)
    return np.stack((zeros, zeros, np.sin(half), np.cos(half)), axis=-1)


def _max_local_observation(
    body_pos: np.ndarray,
    body_quat_xyzw: np.ndarray,
    body_lin_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> np.ndarray:
    """NumPy equivalent of the body-count-dependent max-local observation."""
    root_pos = body_pos[0]
    heading_inv = _heading_inverse(body_quat_xyzw[0])
    heading_all = np.broadcast_to(heading_inv, body_quat_xyzw.shape)

    local_pos = _quat_rotate(heading_all, body_pos - root_pos)[1:].reshape(-1)
    local_quat = _quat_mul(heading_all, body_quat_xyzw)
    unit_x = np.broadcast_to(np.array([1.0, 0.0, 0.0]), body_pos.shape)
    unit_z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), body_pos.shape)
    tan_norm = np.concatenate((_quat_rotate(local_quat, unit_x), _quat_rotate(local_quat, unit_z)), axis=-1).reshape(-1)
    local_vel = _quat_rotate(heading_all, body_lin_vel).reshape(-1)
    local_ang_vel = _quat_rotate(heading_all, body_ang_vel).reshape(-1)
    return np.concatenate(([root_pos[2]], local_pos, tan_norm, local_vel, local_ang_vel)).astype(np.float32)


def _prefix_reference_tree(root_body: ET.Element) -> ET.Element:
    reference = copy.deepcopy(root_body)
    for element in reference.iter():
        name = element.get("name")
        if name:
            element.set("name", f"reference_{name}")
        if element.tag == "body":
            element.set("gravcomp", "1")
        elif element.tag == "geom":
            element.set("contype", "0")
            element.set("conaffinity", "0")
            element.set("group", "1")
            element.set("rgba", "0.15 0.85 0.25 0.38")
    return reference


def _xml_body_names(xml_path: Path) -> list[str]:
    worldbody = ET.parse(xml_path).getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {xml_path}")
    return [str(body.get("name")) for body in worldbody.iter("body") if body.get("name")]


def _load_scene(scene_path: Path, robot_xml_path: Path, timestep: float, show_reference: bool) -> mujoco.MjModel:
    if not show_reference:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.opt.timestep = timestep
        return model

    scene_root = ET.parse(scene_path).getroot()
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene MJCF has no <worldbody>: {scene_path}")

    robot_worldbody = ET.parse(robot_xml_path).getroot().find("worldbody")
    if robot_worldbody is None:
        raise ValueError(f"Robot MJCF has no <worldbody>: {robot_xml_path}")
    robot_roots = robot_worldbody.findall("body")
    if len(robot_roots) != 1:
        raise ValueError(f"Expected one root robot body in {robot_xml_path}, found {len(robot_roots)}")
    worldbody.append(_prefix_reference_tree(robot_roots[0]))

    # Keep the temporary scene beside the real scene so relative <include>
    # and mesh paths retain normal MuJoCo resolution semantics.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".onnx_mujoco_scene_",
            dir=scene_path.parent,
            delete=False,
            encoding="utf-8",
        ) as temporary:
            temporary.write(ET.tostring(scene_root, encoding="unicode"))
            temporary_path = Path(temporary.name)
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    model.opt.timestep = timestep
    return model


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, obj, index)
    if value is None:
        raise ValueError(f"Unnamed {obj} at index {index}")
    return value


def _load_motion(path: Path, joint_names: list[str]) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as motion:
        required = {"fps", "joint_pos", "joint_names", "body_pos_w", "body_quat_w", "body_names"}
        missing = sorted(required.difference(motion.files))
        if missing:
            raise ValueError(f"Motion {path} is missing fields: {missing}")
        source_joints = [str(x) for x in motion["joint_names"].tolist()]
        body_names = [str(x) for x in motion["body_names"].tolist()]
        if "base_link" not in body_names:
            raise ValueError(f"Motion {path} has no base_link body")
        joint_index = {name: i for i, name in enumerate(source_joints)}
        absent = [name for name in joint_names if name not in joint_index]
        if absent:
            raise ValueError(f"Motion {path} is missing joints: {absent}")
        base_index = body_names.index("base_link")
        root_pos = np.asarray(motion["body_pos_w"][:, base_index], dtype=np.float64)
        # BFM/IsaacLab named NPZ stores body_quat_w in wxyz order.
        root_quat_wxyz = np.asarray(motion["body_quat_w"][:, base_index], dtype=np.float64)
        joints = np.asarray(motion["joint_pos"][:, [joint_index[name] for name in joint_names]], dtype=np.float64)
        qpos = np.concatenate((root_pos, root_quat_wxyz, joints), axis=1)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if qpos.shape[0] < 2 or fps <= 0:
        raise ValueError(f"Motion must have at least two frames and positive fps, got shape={qpos.shape}, fps={fps}")
    return qpos, fps


def _reference_kinematics(
    model: mujoco.MjModel,
    qpos_frames: np.ndarray,
    body_ids: np.ndarray,
    frame: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one reference frame and its finite-difference velocity."""
    data = mujoco.MjData(model)
    frame %= len(qpos_frames)
    prev_frame = (frame - 1) % len(qpos_frames)
    next_frame = (frame + 1) % len(qpos_frames)
    data.qpos[:] = qpos_frames[frame]
    if frame == 0:
        mujoco.mj_differentiatePos(model, data.qvel, 1.0 / fps, qpos_frames[frame], qpos_frames[next_frame])
    elif frame == len(qpos_frames) - 1:
        mujoco.mj_differentiatePos(model, data.qvel, 1.0 / fps, qpos_frames[prev_frame], qpos_frames[frame])
    else:
        mujoco.mj_differentiatePos(model, data.qvel, 2.0 / fps, qpos_frames[prev_frame], qpos_frames[next_frame])
    mujoco.mj_forward(model, data)

    body_pos = data.xpos[body_ids].copy()
    body_quat_xyzw = data.xquat[body_ids][:, [1, 2, 3, 0]].copy()
    body_lin_vel = np.empty((len(body_ids), 3), dtype=np.float64)
    body_ang_vel = np.empty_like(body_lin_vel)
    spatial = np.empty(6, dtype=np.float64)
    for out_index, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), spatial, 0)
        body_ang_vel[out_index] = spatial[:3]
        body_lin_vel[out_index] = spatial[3:]
    return body_pos, body_quat_xyzw, body_lin_vel, body_ang_vel


def _reference_encoder_inputs(
    model: mujoco.MjModel,
    qpos_frames: np.ndarray,
    body_ids: np.ndarray,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    frame: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    body_pos, body_quat, body_vel, body_ang_vel = _reference_kinematics(model, qpos_frames, body_ids, frame, fps)
    ref_qpos = qpos_frames[frame % len(qpos_frames)]
    qvel = np.empty(model.nv, dtype=np.float64)
    next_frame = min(frame % len(qpos_frames) + 1, len(qpos_frames) - 1)
    prev_frame = max(frame % len(qpos_frames) - 1, 0)
    span = max(next_frame - prev_frame, 1) / fps
    mujoco.mj_differentiatePos(model, qvel, span, qpos_frames[prev_frame], qpos_frames[next_frame])
    dof_pos = ref_qpos[qpos_adr].astype(np.float32)
    dof_vel = qvel[dof_adr].astype(np.float32)
    projected_gravity = _quat_rotate_inverse(body_quat[0], np.array([0.0, 0.0, -1.0])).astype(np.float32)
    ref_ang_vel = (0.25 * _quat_rotate_inverse(body_quat[0], body_ang_vel[0])).astype(np.float32)
    state = np.concatenate((dof_pos, dof_vel, projected_gravity, ref_ang_vel)).astype(np.float32)
    privileged_state = _max_local_observation(body_pos, body_quat, body_vel, body_ang_vel)
    return state[None], privileged_state[None]


def _current_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_adr: np.ndarray,
    joint_dof_adr: np.ndarray,
    base_body_id: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dof_pos = data.qpos[joint_qpos_adr].astype(np.float32).copy()
    dof_vel = data.qvel[joint_dof_adr].astype(np.float32).copy()
    base_quat = data.xquat[base_body_id, [1, 2, 3, 0]].copy()
    projected_gravity = _quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0])).astype(np.float32)
    spatial = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, spatial, 0)
    base_ang_vel = (0.25 * _quat_rotate_inverse(base_quat, spatial[:3])).astype(np.float32)
    state = np.concatenate((dof_pos, dof_vel, projected_gravity, base_ang_vel)).astype(np.float32)
    return state, {
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "projected_gravity": projected_gravity,
    }


def _config_values(
    config_path: Path, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    config = yaml.safe_load(config_path.read_text())
    if "training" in config:
        robot = config["training"]
        config_joint_names = list(config["control_joints"]["names"])
        effort_key = "effort_limit"
        actuator_joints = robot.get("actuator", {}).get("joints", {})
        armature = np.array([float(actuator_joints[name]["armature"]) for name in joint_names], dtype=np.float64)
        friction = np.array([float(actuator_joints[name]["friction"]) for name in joint_names], dtype=np.float64)
    else:
        robot = config["robot"]
        config_joint_names = list(robot["dof_names"])
        effort_key = "dof_effort_limit_list"
        armature = None
        friction = None
    control = robot["control"]
    kp = np.array([float(control["stiffness"][name]) for name in joint_names], dtype=np.float64)
    kd = np.array([float(control["damping"][name]) for name in joint_names], dtype=np.float64)
    effort = np.asarray(control[effort_key] if effort_key == "effort_limit" else robot[effort_key], dtype=np.float64)
    if config_joint_names != joint_names:
        index = {name: i for i, name in enumerate(config_joint_names)}
        effort = effort[[index[name] for name in joint_names]]
    scale = np.full(len(joint_names), float(control["action_scale"]), dtype=np.float64)
    if bool(control.get("action_rescale", True)):
        scale *= effort / kp
    return kp, kd, effort, scale, armature, friction


def _ort_session(ort, path: Path, provider: str, intra_op_threads: int = 0):
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    session_options = ort.SessionOptions()
    if intra_op_threads > 0:
        session_options.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=[requested])
    if requested not in session.get_providers():
        raise RuntimeError(f"ONNX Runtime could not enable {requested}; active providers={session.get_providers()}")
    return session


def _inference_worker(connection, policy_path: str, provider: str, intra_op_threads: int) -> None:
    try:
        import onnxruntime as ort

        if provider == "cuda":
            ort.preload_dlls()
        policy = _ort_session(ort, Path(policy_path), provider, intra_op_threads)
        connection.send(("ready", policy.get_providers()[0]))
        while True:
            request = connection.recv()
            if request is None:
                break
            sequence, actor_obs = request
            infer_start = time.perf_counter()
            raw_action = policy.run(None, {"actor_obs": actor_obs})[0][0]
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            connection.send(("result", sequence, raw_action, infer_ms))
    except BaseException as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _measure_onnx(session, output_names, inputs: dict[str, np.ndarray], warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        session.run(output_names, inputs)
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(output_names, inputs)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations, iterations / elapsed


def _benchmark_onnx(ort, exported: Path, warmup: int, iterations: int) -> None:
    available = set(ort.get_available_providers())
    providers = ["cpu"]
    if "CUDAExecutionProvider" in available:
        # Load CUDA/cuDNN libraries provided by this uv environment.
        ort.preload_dlls()
        providers.append("cuda")

    rng = np.random.default_rng(0)
    state = rng.standard_normal((1, 48), dtype=np.float32)
    privileged_state = rng.standard_normal((1, 358), dtype=np.float32)
    actor_obs = rng.standard_normal((1, 601), dtype=np.float32)

    print(f"[onnx-benchmark] warmup={warmup}, iterations={iterations}, batch=1")
    print("[onnx-benchmark] synchronous ORT run, including CPU<->GPU copies")
    print("[onnx-benchmark] current MuJoCo playback uses precomputed z, so actor timing is its actual ONNX cost")
    print(f"{'provider':<10} {'backward ms':>12} {'backward Hz':>12} {'actor ms':>12} {'actor Hz':>12} {'chain ms':>12} {'chain Hz':>12}")
    results: dict[str, tuple[float, float, float]] = {}
    for provider in providers:
        backward = _ort_session(ort, exported / "backward_encoder.onnx", provider)
        policy = _ort_session(ort, exported / "FBcprAuxModel.onnx", provider)
        backward_inputs = {"state": state, "privileged_state": privileged_state}
        backward_ms, backward_hz = _measure_onnx(backward, ["z"], backward_inputs, warmup, iterations)
        actor_ms, actor_hz = _measure_onnx(policy, ["action"], {"actor_obs": actor_obs}, warmup, iterations)

        for _ in range(warmup):
            z = backward.run(["z"], backward_inputs)[0]
            actor_obs[:, -z.shape[1] :] = z
            policy.run(["action"], {"actor_obs": actor_obs})
        start = time.perf_counter()
        for _ in range(iterations):
            z = backward.run(["z"], backward_inputs)[0]
            actor_obs[:, -z.shape[1] :] = z
            policy.run(["action"], {"actor_obs": actor_obs})
        elapsed = time.perf_counter() - start
        chain_ms = elapsed * 1000.0 / iterations
        chain_hz = iterations / elapsed
        results[provider] = (backward_ms, actor_ms, chain_ms)
        print(
            f"{provider:<10} {backward_ms:12.4f} {backward_hz:12.1f} "
            f"{actor_ms:12.4f} {actor_hz:12.1f} {chain_ms:12.4f} {chain_hz:12.1f}"
        )

    if "cuda" not in results:
        print(f"[onnx-benchmark] CUDA unavailable; available providers={ort.get_available_providers()}")
    else:
        cpu = results["cpu"]
        cuda = results["cuda"]
        print(
            "[onnx-benchmark] CUDA speedup over CPU: "
            f"backward={cpu[0] / cuda[0]:.2f}x, actor={cpu[1] / cuda[1]:.2f}x, chain={cpu[2] / cuda[2]:.2f}x"
        )


def run(args: argparse.Namespace) -> None:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit("onnxruntime is required; install it with: uv add onnxruntime") from exc

    model_folder = args.model_folder.expanduser().resolve()
    exported = model_folder / "exported"
    meta_path = exported / "FBcprAuxModel.meta.json"
    meta = json.loads(meta_path.read_text())
    runtime_metadata_path = model_folder / "tracking_inference" / "metadata.json"
    runtime_metadata = json.loads(runtime_metadata_path.read_text(encoding="utf-8"))
    if (
        runtime_metadata.get("schema_version") != 1
        or runtime_metadata.get("format") != "ufo_precomputed_onnx_latent"
    ):
        raise ValueError(f"Unsupported runtime metadata: {runtime_metadata_path}")
    runtime_dir = runtime_metadata_path.parent
    control_metadata = runtime_metadata["control"]
    latent_metadata = runtime_metadata["latent"]
    actor_path = (runtime_dir / runtime_metadata["actor"]["file"]).resolve()
    joint_names = [str(x) for x in control_metadata["joint_names"]]
    if joint_names != [str(x) for x in meta["control_joint_names"]]:
        raise ValueError("Runtime metadata and Actor export joint orders differ")

    if args.benchmark:
        _benchmark_onnx(ort, exported, args.benchmark_warmup, args.benchmark_iterations)
        return

    xml_path = _resolve_metadata_path(args.xml, str(meta["xml_path"]), "Robot XML")
    scene_path = args.scene.expanduser().resolve()
    config_path = _resolve_metadata_path(
        args.robot_config, str(meta["robot_config_path"]), "Robot config"
    )
    model = _load_scene(
        scene_path,
        robot_xml_path=xml_path,
        timestep=1.0 / args.sim_fps,
        show_reference=not args.no_reference,
    )
    reference_model = mujoco.MjModel.from_xml_path(str(xml_path))
    reference_model.opt.timestep = 1.0 / args.sim_fps
    data = mujoco.MjData(model)

    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    if np.any(joint_ids < 0):
        missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        raise ValueError(f"MJCF is missing policy joints: {missing}")
    joint_qpos_adr = model.jnt_qposadr[joint_ids].astype(int)
    joint_dof_adr = model.jnt_dofadr[joint_ids].astype(int)

    actuator_by_joint = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)}
    actuator_ids = np.array([actuator_by_joint[int(joint_id)] for joint_id in joint_ids], dtype=int)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    motion_path = args.motion.expanduser().resolve()
    expected_motion_sha = runtime_metadata.get("source", {}).get("motion_sha256")
    if expected_motion_sha and _sha256(motion_path) != expected_motion_sha:
        raise ValueError(
            "The selected NPZ does not match the latent metadata. "
            "Regenerate z with run_onnx_mujoco.sh or humanoidverse.generate_onnx_latent."
        )
    qpos_frames, motion_fps = _load_motion(motion_path, joint_names)
    if qpos_frames.shape[1] != reference_model.nq:
        raise ValueError(f"Motion qpos width {qpos_frames.shape[1]} does not match robot MJCF nq={reference_model.nq}")
    _config_kp, _config_kd, _config_effort, _config_scale, armature, friction = _config_values(
        config_path, joint_names
    )
    kp = np.asarray(control_metadata["p_gains"], dtype=np.float64).copy()
    kd = np.asarray(control_metadata["d_gains"], dtype=np.float64).copy()

    effort = np.asarray(control_metadata["effort_limits"], dtype=np.float64)
    action_target_scale = np.asarray(control_metadata["target_scales"], dtype=np.float64)
    action_obs_scale = float(control_metadata["action_obs_scale"])
    action_clip = float(control_metadata["action_clip"])
    if armature is not None:
        model.dof_armature[joint_dof_adr] = armature
    if friction is not None:
        model.dof_frictionloss[joint_dof_adr] = friction

    # The scene supplies ground contact, while the robot YAML supplies passive
    # actuator dynamics.  Do not overwrite them with values from an unrelated
    # deployment adapter.
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("MuJoCo scene has no floor geom")

    reference_free_qpos_adr: int | None = None
    reference_joint_qpos_adr: np.ndarray | None = None
    if not args.no_reference:
        free_joint_ids = np.flatnonzero(reference_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one free joint in robot MJCF, found {len(free_joint_ids)}")
        free_joint_name = _name(reference_model, mujoco.mjtObj.mjOBJ_JOINT, int(free_joint_ids[0]))
        reference_free_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"reference_{free_joint_name}"
        )
        if reference_free_joint_id < 0:
            raise ValueError(f"Duplicated reference robot has no reference_{free_joint_name} free joint")
        reference_free_qpos_adr = int(model.jnt_qposadr[reference_free_joint_id])
        reference_joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"reference_{name}") for name in joint_names]
        )
        reference_joint_qpos_adr = model.jnt_qposadr[reference_joint_ids].astype(int)

    policy = None
    inference_connection = None
    inference_process = None
    if args.inference_mode == "inline":
        if args.onnx_provider == "cuda":
            ort.preload_dlls()
        policy = _ort_session(ort, actor_path, args.onnx_provider, args.onnx_threads)
        active_provider = policy.get_providers()[0]
    else:
        context = mp.get_context("spawn")
        inference_connection, child_connection = context.Pipe()
        inference_process = context.Process(
            target=_inference_worker,
            args=(child_connection, str(actor_path), args.onnx_provider, args.onnx_threads),
            name="ufo-onnx-inference",
        )
        inference_process.start()
        child_connection.close()
        ready = inference_connection.recv()
        if ready[0] != "ready":
            raise RuntimeError(f"Inference worker failed during startup: {ready}")
        active_provider = ready[1]
    latent_path = (
        args.latent
        if args.latent is not None
        else runtime_dir / latent_metadata["file"]
    ).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Precomputed latent file does not exist: {latent_path}")
    latents = np.load(latent_path, allow_pickle=False)
    if latents.dtype != np.float32:
        raise ValueError(f"Expected float32 latent, got {latents.dtype} from {latent_path}")
    expected_z_dim = int(latent_metadata["shape"][1])
    if latents.ndim != 2 or latents.shape[1] != expected_z_dim:
        raise ValueError(
            f"Expected latent shape [frames, {expected_z_dim}], got {latents.shape} "
            f"from {latent_path}"
        )
    if len(latents) == 0 or not np.isfinite(latents).all():
        raise ValueError(f"Precomputed latent is empty or non-finite: {latent_path}")
    playback_frame_count = min(len(qpos_frames), len(latents))

    data.qpos[: reference_model.nq] = qpos_frames[0]
    if reference_free_qpos_adr is not None and reference_joint_qpos_adr is not None:
        data.qpos[reference_free_qpos_adr : reference_free_qpos_adr + 7] = qpos_frames[0, :7]
        data.qpos[reference_free_qpos_adr + 1] += args.reference_offset_y
        data.qpos[reference_joint_qpos_adr] = qpos_frames[0, 7:]
    mujoco.mj_forward(model, data)
    history = {
        "actions": np.zeros((4, len(joint_names)), dtype=np.float32),
        "base_ang_vel": np.zeros((4, 3), dtype=np.float32),
        "dof_pos": np.zeros((4, len(joint_names)), dtype=np.float32),
        "dof_vel": np.zeros((4, len(joint_names)), dtype=np.float32),
        "projected_gravity": np.zeros((4, 3), dtype=np.float32),
    }
    action = np.zeros(len(joint_names), dtype=np.float32)
    control_dt = args.decimation / args.sim_fps
    max_steps = args.max_steps if args.max_steps > 0 else math.inf

    print(f"[onnx-mujoco] policy={actor_path}")
    print(f"[onnx-mujoco] runtime_metadata={runtime_metadata_path}")
    print(
        f"[onnx-mujoco] inference_mode={args.inference_mode}, onnx_provider={active_provider}, "
        f"onnx_threads={args.onnx_threads or 'default'}"
    )
    print(f"[onnx-mujoco] latent={latent_path} ({len(latents)} frames x {latents.shape[1]})")
    print(f"[onnx-mujoco] xml={xml_path}")
    print(f"[onnx-mujoco] scene={scene_path}")
    print(f"[onnx-mujoco] motion={args.motion.resolve()} ({len(qpos_frames)} frames @ {motion_fps:g} Hz)")
    arm_names = [name for name in joint_names if name.startswith("zarm_")]
    arm_pd = ", ".join(
        f"{name}:Kp={kp[joint_names.index(name)]:.4f}/Kd={kd[joint_names.index(name)]:.4f}"
        for name in arm_names
    )
    print(f"[onnx-mujoco] arm_pd={arm_pd}")
    if len(qpos_frames) != len(latents):
        print(
            f"[onnx-mujoco] playback uses {playback_frame_count} frames "
            f"(motion={len(qpos_frames)}, latent={len(latents)})"
        )
    print(f"[onnx-mujoco] sim={args.sim_fps:g} Hz, policy={1.0 / control_dt:g} Hz, headless={args.headless}")

    def simulate(viewer=None) -> None:
        nonlocal action
        step = 0
        start = time.perf_counter()
        inference_samples: list[float] = []
        roundtrip_samples: list[float] = []
        total_inference_samples = 0
        while step < max_steps and (viewer is None or viewer.is_running()):
            tick = time.perf_counter()
            motion_frame = int(round(step * control_dt * motion_fps))
            if not args.loop and motion_frame >= playback_frame_count:
                break
            motion_frame %= playback_frame_count
            # motion_frame = 0 #TODO
            state, current = _current_observation(model, data, joint_qpos_adr, joint_dof_adr, base_body_id)
            history_actor = np.concatenate([history[key].reshape(-1) for key in sorted(history)]).astype(np.float32)
            z = latents[motion_frame]
            actor_obs = np.concatenate((state, action, history_actor, z)).astype(np.float32)[None]
            roundtrip_start = time.perf_counter()
            if args.inference_mode == "inline":
                infer_start = time.perf_counter()
                raw_action = policy.run(None, {"actor_obs": actor_obs})[0][0]
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
            else:
                inference_connection.send((step, actor_obs))
                response = inference_connection.recv()
                if response[0] == "error":
                    raise RuntimeError(f"Inference worker failed: {response[1]}")
                if response[0] != "result" or response[1] != step:
                    raise RuntimeError(f"Unexpected inference response: {response[:2]}")
                raw_action = response[2]
                infer_ms = float(response[3])
            roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000.0
            inference_samples.append(infer_ms)
            roundtrip_samples.append(roundtrip_ms)
            total_inference_samples += 1

            if len(inference_samples) >= args.latency_log_every:
                print(
                    "[onnx-mujoco] Actor inference latency: "
                    f"last={inference_samples[-1]:.3f} ms, avg={np.mean(inference_samples):.3f} ms, "
                    f"max={np.max(inference_samples):.3f} ms, samples={len(inference_samples)}, "
                    f"total_samples={total_inference_samples}"
                )
                if args.inference_mode == "process":
                    print(
                        "[onnx-mujoco] Actor IPC roundtrip latency: "
                        f"last={roundtrip_samples[-1]:.3f} ms, avg={np.mean(roundtrip_samples):.3f} ms, "
                        f"max={np.max(roundtrip_samples):.3f} ms, samples={len(roundtrip_samples)}, "
                        f"total_samples={total_inference_samples}"
                    )
                inference_samples.clear()
                roundtrip_samples.clear()

            # Match the training environment's observation timing: history is
            # queried before the current observation is inserted, so its newest
            # action is one policy step older than ``last_action``.  Insert the
            # action used in actor_obs before replacing it with the new output.
            for key, value in (("actions", action), *current.items()):
                history[key][1:] = history[key][:-1]
                history[key][0] = value

            action = np.clip(
                raw_action * action_obs_scale, -action_clip, action_clip
            ).astype(np.float32)
            target = action * action_target_scale
            for _ in range(args.decimation):
                torque = kp * (target - data.qpos[joint_qpos_adr]) - kd * data.qvel[joint_dof_adr]
                data.ctrl[actuator_ids] = np.clip(torque, -effort, effort)
                mujoco.mj_step(model, data)

            if reference_free_qpos_adr is not None and reference_joint_qpos_adr is not None:
                data.qpos[reference_free_qpos_adr : reference_free_qpos_adr + 7] = qpos_frames[motion_frame, :7]
                data.qpos[reference_free_qpos_adr + 1] += args.reference_offset_y
                data.qpos[reference_joint_qpos_adr] = qpos_frames[motion_frame, 7:]
                mujoco.mj_forward(model, data)

            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"Simulation became non-finite at policy step {step}")
            if viewer is not None:
                viewer.sync()
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                speed = step / max(time.perf_counter() - start, 1e-6)
                print(
                    f"[onnx-mujoco] step={step} motion_frame={motion_frame} base_z={data.qpos[2]:.3f} "
                    f"action=[{action.min():.2f}, {action.max():.2f}] speed={speed:.1f} policy_steps/s"
                )

    try:
        if args.headless:
            simulate()
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                # viewer.cam.trackbodyid = base_body_id
                # viewer.cam.distance = 3.0
                # viewer.cam.azimuth = 135.0
                # viewer.cam.elevation = -18.0
                simulate(viewer)
    finally:
        if inference_connection is not None:
            try:
                inference_connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
            inference_connection.close()
        if inference_process is not None:
            inference_process.join(timeout=5.0)
            if inference_process.is_alive():
                inference_process.terminate()
                inference_process.join(timeout=1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_RUN, help="Run folder containing exported/*.onnx")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="RobotState NPZ motion")
    parser.add_argument(
        "--latent",
        type=Path,
        default=None,
        help="Precomputed .npy latent; defaults to <model-folder>/tracking_inference/zs_0.npy",
    )
    parser.add_argument("--xml", type=Path, default=None, help="Override MJCF path from export metadata")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MuJoCo scene XML containing the robot and world")
    parser.add_argument("--robot-config", type=Path, default=None, help="Override robot YAML path from export metadata")
    parser.add_argument("--sim-fps", type=float, default=200.0)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--onnx-threads", type=int, default=0, help="ORT intra-op threads; 0 uses the ORT default")
    parser.add_argument("--inference-mode", choices=("inline", "process"), default="inline")
    parser.add_argument("--latency-log-every", type=int, default=100)
    parser.add_argument("--benchmark", action="store_true", help="Benchmark CPU and CUDA ONNX inference, then exit")
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0, help="Policy steps; 0 runs until the window closes")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-reference", action="store_true", help="Hide the translucent reference robot")
    parser.add_argument("--reference-offset-y", type=float, default=1.2, help="Side-by-side reference robot offset")
    args = parser.parse_args()
    if (
        args.sim_fps <= 0
        or args.decimation <= 0
        or args.onnx_threads < 0
        or args.latency_log_every <= 0
        or args.benchmark_warmup < 0
        or args.benchmark_iterations <= 0
    ):
        parser.error("fps, decimation and benchmark iterations must be positive; benchmark warmup cannot be negative")
    return args


if __name__ == "__main__":
    run(_parse_args())

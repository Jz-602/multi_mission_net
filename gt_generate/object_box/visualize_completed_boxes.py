"""原始框绿色、消失后外推橙色、首次出现前回推紫色；支持Q/E逐帧、空格播放。

python gt_generate/object_box/visualize_completed_boxes.py --data-root "$WAYMO_DATA_ROOT"
增加 --frame 40 --preview /tmp/completed_boxes.png 可生成单帧俯视图，无需桌面。
全部补框来自内存；地面使用全局图转换，避免依赖尚未统一重导出的逐帧地面。
"""

from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gt_generate.common.io_utils import LoadPcdXyzi
from gt_generate.ground_truth.data_paths import BuildTimestampFile
from gt_generate.object_box.complete_detection_boxes import BuildParser, CompleteDetectionBoxes, PrintSummary, RunExperiment, TYPE_NAMES


ORIGINAL_COLOR = (0.15, 1.0, 0.25)
ADDED_COLOR = (1.0, 0.48, 0.03)
BACKWARD_COLOR = (0.76, 0.35, 1.0)
ROAD_COLOR = (0.06, 0.28, 0.55)
LOW_GROUND_COLOR = (0.08, 0.46, 0.45)
BOX_EDGES = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                        [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int32)


def BoxCorners(box):
    """box [7] 和输出 [8,3] 顶点都在当前帧 LiDAR 系，框只包含 yaw。"""
    corners = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    corners *= np.asarray(box[3:6]) / 2.0
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rotation.T + np.asarray(box[:3])


def LoadFrameData(result, frame_idx, max_distance_m):
    frame = result["frames"][frame_idx]
    timestamp = int(frame["frame_timestamp_micros"]) // 1000
    output = result["sequence_root"] / "output"
    pose = np.asarray(frame["pose"], dtype=np.float64)
    # 全局地面 [N,3] 由世界系变回当前帧 LiDAR 系，使用完整旋转和平移。
    world_ground, ground_types = result["ground_points"]
    local_ground = (world_ground - pose[:3, 3]) @ pose[:3, :3]
    b_ground = np.linalg.norm(local_ground[:, :2], axis=1) <= max_distance_m
    local_ground, ground_types = local_ground[b_ground], ground_types[b_ground]
    ground_colors = np.asarray([ROAD_COLOR if value == 1 else LOW_GROUND_COLOR for value in ground_types]).reshape(-1, 3)

    raw = LoadPcdXyzi(BuildTimestampFile(str(output / "pcd"), timestamp, ".pcd"))[::3, :3]
    raw = raw[np.linalg.norm(raw[:, :2], axis=1) <= max_distance_m]
    occ_path = Path(BuildTimestampFile(str(output / "label/background_occ"), timestamp, ".npz"))
    occ = np.empty((0, 3), dtype=np.float64)
    if occ_path.exists():
        with np.load(occ_path, allow_pickle=False) as data:
            # sparse OCC 的索引与体素大小均定义在当前帧 LiDAR 系。
            occ = data["pc_range"][:3] + (data["voxel_indices"][::2].astype(np.float64) + 0.5) * data["voxel_size_xyz"]
            occ = occ[np.linalg.norm(occ[:, :2], axis=1) <= max_distance_m]
    originals = [obj for obj in frame["objects"] if int(obj["type"]) in TYPE_NAMES
                 and np.linalg.norm(np.asarray(obj["box"])[:2]) <= max_distance_m]
    stops = [event for event in result["events"] if event["stop_frame"] == frame_idx]
    return {"timestamp": timestamp, "ground": local_ground, "ground_colors": ground_colors,
            "raw": raw, "occ": occ, "originals": originals,
            "added": result["supplements"][frame_idx], "stops": stops,
            "forward": [obj for obj in result["supplements"][frame_idx] if obj["direction"] == "forward"],
            "backward": [obj for obj in result["supplements"][frame_idx] if obj["direction"] == "backward"]}


def SavePreview(result, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = LoadFrameData(result, args.frame, args.max_distance_m)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#101824")
    ax.set_facecolor("#101824")
    ax.scatter(data["ground"][:, 0], data["ground"][:, 1], c=data["ground_colors"], s=2, linewidths=0)
    ax.scatter(data["occ"][:, 0], data["occ"][:, 1], color="#727982", s=0.4, linewidths=0, alpha=0.65)
    for objects, color in [(data["originals"], ORIGINAL_COLOR), (data["forward"], ADDED_COLOR), (data["backward"], BACKWARD_COLOR)]:
        for obj in objects:
            corners = BoxCorners(obj["box"])[[0, 1, 2, 3, 0]]
            ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.5)
            if "event_index" in obj:
                ax.text(obj["box"][0], obj["box"][1], f"+{obj['event_index']}", color=color, fontsize=7)
    circle = plt.Circle((0, 0), args.max_distance_m, color="#bdd0f5", fill=False, linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.scatter([0], [0], color="white", marker="^", s=55)
    ax.set(xlim=(-args.max_distance_m - 3, args.max_distance_m + 3), ylim=(-args.max_distance_m - 3, args.max_distance_m + 3),
           xlabel="Current LiDAR x (m)", ylabel="Current LiDAR y (m)")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_title(f"{args.sequence}\nFrame {args.frame}/{len(result['frames']) - 1} | {result['motion_model']} | Anchor limit {args.max_anchor_distance_m:g}m\nOriginal {len(data['originals'])} | Forward {len(data['forward'])} | Backward {len(data['backward'])}", color="white", fontsize=11)
    legend = [Line2D([], [], color=color, label=label, linewidth=2) for color, label in
              [(ORIGINAL_COLOR, "Original boxes"), (ADDED_COLOR, "After disappearance"), (BACKWARD_COLOR, "Before first appearance"),
               (ROAD_COLOR, "ROAD support"), (LOW_GROUND_COLOR, "LOW_GROUND support")]]
    ax.legend(handles=legend, loc="lower left", facecolor="#172234", labelcolor="white")
    fig.tight_layout()
    preview_path = Path(args.preview).expanduser()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Preview: {preview_path}")


def ShowExperiment(result, args):
    import open3d as o3d
    from open3d.visualization import gui, rendering

    application = gui.Application.instance
    application.initialize()
    window = application.create_window(f"Detection box recovery | {args.max_distance_m:g}m | anchor {args.max_anchor_distance_m:g}m", 1600, 950)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background(np.asarray([0.06, 0.09, 0.14, 1.0]))
    panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))
    panel.add_child(gui.Label("Original: GREEN | Forward: ORANGE\nBackward: PURPLE"))
    panel.add_child(gui.Label("ROAD: blue | LOW_GROUND: teal"))
    panel.add_child(gui.Label("Q/E: previous/next | Space: play/pause"))
    model_selector = gui.Combobox()
    model_selector.add_item("Velocity: straight extrapolation")
    model_selector.add_item(f"Acceleration: {args.history_frames} observed frames")
    model_selector.selected_index = 1 if args.motion_model == "acceleration" else 0
    panel.add_child(model_selector)
    panel.add_child(gui.Label(f"Range: {args.max_distance_m:g}m | Map support only"))
    panel.add_child(gui.Label(f"Distance from original anchor <= {args.max_anchor_distance_m:g}m"))
    status = gui.Label("")
    panel.add_child(status)
    slider = gui.Slider(gui.Slider.INT)
    slider.set_limits(0, len(result["frames"]) - 1)
    panel.add_child(slider)
    controls = gui.Horiz(6)
    previous = gui.Button("Previous")
    play = gui.Button("Play / Pause")
    following = gui.Button("Next")
    for button in (previous, play, following):
        controls.add_child(button)
    panel.add_child(controls)
    checks = {}
    for key, label, b_checked in [("raw", "Current point cloud", False), ("ground", "Global ground support", True),
                                  ("occ", "Static OCC (context only)", True), ("originals", "Original boxes", True),
                                  ("forward", "After disappearance", True), ("backward", "Before first appearance", True)]:
        check = gui.Checkbox(label)
        check.checked = b_checked
        checks[key] = check
        panel.add_child(check)
    detail = gui.Label("")
    panel.add_child(detail)
    panel.add_child(gui.Label("Stop when the path leaves map support.\nOCC does not stop extrapolation in this version."))
    window.add_child(scene)
    window.add_child(panel)
    state = {"frame": args.frame, "b_dirty": True, "b_paused": True, "last_time": time.monotonic()}
    point_material = rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = 2.0
    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 2.5

    def SetFrame(frame_idx):
        state.update(frame=max(0, min(len(result["frames"]) - 1, int(frame_idx))), b_dirty=True, b_paused=True)

    def TogglePlay():
        state["b_paused"] = not state["b_paused"]
        state["last_time"] = time.monotonic()

    def MarkDirty(_):
        state["b_dirty"] = True

    def SetMotionModel(_, selected_index):
        args.motion_model = "acceleration" if selected_index == 1 else "velocity"
        frames, supplements, events = CompleteDetectionBoxes(
            result["frames"], result["terrain"], args.max_distance_m,
            args.motion_model, args.history_frames, args.max_acceleration_mps2, args.max_anchor_distance_m,
        )
        result.update(frames=frames, supplements=supplements, events=events, motion_model=args.motion_model)
        state.update(b_dirty=True, b_paused=True)
        PrintSummary(result)

    model_selector.set_on_selection_changed(SetMotionModel)
    slider.set_on_value_changed(SetFrame)
    previous.set_on_clicked(lambda: SetFrame(state["frame"] - 1))
    following.set_on_clicked(lambda: SetFrame(state["frame"] + 1))
    play.set_on_clicked(TogglePlay)
    for check in checks.values():
        check.set_on_checked(MarkDirty)

    def OnLayout(_):
        rect = window.content_rect
        panel_width = 390
        panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        scene.frame = gui.Rect(rect.x + panel_width, rect.y, rect.width - panel_width, rect.height)

    def OnKey(event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == gui.KeyName.Q:
            SetFrame(state["frame"] - 1)
        elif event.key == gui.KeyName.E:
            SetFrame(state["frame"] + 1)
        elif event.key == gui.KeyName.SPACE:
            TogglePlay()
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def AddPoints(name, points, colors):
        if not len(points):
            return
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(np.broadcast_to(colors, points.shape).copy())
        scene.scene.add_geometry(name, cloud, point_material)

    def AddBoxes(name, objects, color):
        if not objects:
            return
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.concatenate([BoxCorners(obj["box"]) for obj in objects]))
        indices = np.concatenate([BOX_EDGES + 8 * i for i in range(len(objects))])
        lines.lines = o3d.utility.Vector2iVector(indices)
        lines.paint_uniform_color(color)
        scene.scene.add_geometry(name, lines, line_material)

    def UpdateFrame():
        data = LoadFrameData(result, state["frame"], args.max_distance_m)
        scene.scene.clear_geometry()
        for key, colors in [("raw", (0.55, 0.55, 0.55)), ("ground", data["ground_colors"]), ("occ", (0.43, 0.46, 0.5))]:
            if checks[key].checked:
                AddPoints(key, data[key], colors)
        for key, color in [("originals", ORIGINAL_COLOR), ("forward", ADDED_COLOR), ("backward", BACKWARD_COLOR)]:
            if checks[key].checked:
                AddBoxes(key, data[key], color)
        angles = np.linspace(0, 2 * np.pi, 181)
        circle = o3d.geometry.LineSet()
        circle.points = o3d.utility.Vector3dVector(np.column_stack([args.max_distance_m * np.cos(angles), args.max_distance_m * np.sin(angles), np.zeros_like(angles)]))
        circle.lines = o3d.utility.Vector2iVector(np.column_stack([np.arange(180), np.arange(1, 181)]))
        circle.paint_uniform_color((0.65, 0.72, 0.85))
        scene.scene.add_geometry("range", circle, line_material)
        scene.scene.add_geometry("ego", o3d.geometry.TriangleMesh.create_coordinate_frame(size=3), point_material)
        slider.int_value = state["frame"]
        status.text = f"Frame {state['frame']} / {len(result['frames']) - 1}\nOriginal {len(data['originals'])} | Added {len(data['added'])}"
        rows = ["Added boxes: direction, ID, time, distances"]
        for obj in data["added"][:7]:
            rows.append(f"{obj['direction']} {TYPE_NAMES[obj['type']]} {obj['id'][-6:]} {obj['elapsed_s']:+.1f}s\n"
                        f"  ego {obj['distance_m']:.1f}m | anchor {obj['anchor_distance_m']:.1f}m")
        if len(data["added"]) > 7:
            rows.append(f"... {len(data['added']) - 7} more (see terminal)")
        if data["stops"]:
            rows.append("Stops this frame:")
            rows.extend(f"{event['id'][-6:]}: {event['stop_reason']}" for event in data["stops"][:6])
        detail.text = "\n".join(rows)
        print(f"Frame {state['frame']} | added=" + ", ".join(
            f"{obj['direction']} {obj['id']}:{obj['elapsed_s']:+.1f}s/anchor={obj['anchor_distance_m']:.2f}m" for obj in data["added"]), flush=True)
        window.set_needs_layout()
        state.update(b_dirty=False, last_time=time.monotonic())

    def OnTick():
        if not state["b_paused"] and time.monotonic() - state["last_time"] >= 0.1:
            if state["frame"] < len(result["frames"]) - 1:
                state["frame"] += 1
                state["b_dirty"] = True
            else:
                state["b_paused"] = True
        if state["b_dirty"]:
            UpdateFrame()
            return True
        return False

    window.set_on_layout(OnLayout)
    scene.set_on_key(OnKey)
    window.set_on_tick_event(OnTick)
    UpdateFrame()
    bounds = o3d.geometry.AxisAlignedBoundingBox([-args.max_distance_m, -args.max_distance_m, -7], [args.max_distance_m, args.max_distance_m, 6])
    scene.setup_camera(60, bounds, np.zeros(3))
    scene.scene.camera.look_at(np.zeros(3), np.asarray([-45.0, -70.0, 90.0]), np.asarray([0.0, 0.0, 1.0]))
    application.run()


def Main():
    parser = BuildParser(__doc__)
    parser.add_argument("--frame", type=int, default=26, help="初始帧，0 开始")
    parser.add_argument("--preview", default="", help="只保存指定帧的俯视图，不打开窗口")
    args = parser.parse_args()
    result = RunExperiment(args)
    if not 0 <= args.frame < len(result["frames"]):
        parser.error(f"--frame 必须在 0 到 {len(result['frames']) - 1} 之间")
    result["ground_points"] = result["terrain"].WorldPoints()
    PrintSummary(result)
    if args.preview:
        SavePreview(result, args)
    else:
        ShowExperiment(result, args)


if __name__ == "__main__":
    Main()

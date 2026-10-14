import os

import numpy as np


def LoadPoseFile(pose_file_path: str):
    timestamp_list = []
    pose_list = []

    with open(pose_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            timestamp_list.append(int(parts[0]))
            pose_list.append([float(val) for val in parts[1:7]])

    timestamp_array = np.asarray(timestamp_list, dtype=np.int64)
    pose_array = np.asarray(pose_list, dtype=np.float64)
    return timestamp_array, pose_array


def FindLongestLowMotionSegment(
    timestamp_array: np.ndarray,
    pose_array: np.ndarray,
    translation_thresh_meters: float = 0.03,
):
    position_array = pose_array[:, :3]
    delta_position = position_array[1:] - position_array[:-1]
    delta_distance = np.linalg.norm(delta_position, axis=1)
    b_low_motion = delta_distance < translation_thresh_meters

    best_start_idx = -1
    best_end_idx = -1
    best_length = 0

    current_start_idx = -1
    current_length = 0

    for i, b_flag in enumerate(b_low_motion):
        if b_flag:
            if current_start_idx == -1:
                current_start_idx = i
                current_length = 1
            else:
                current_length += 1
        else:
            if current_length > best_length:
                best_start_idx = current_start_idx
                best_end_idx = i
                best_length = current_length
            current_start_idx = -1
            current_length = 0

    if current_length > best_length:
        best_start_idx = current_start_idx
        best_end_idx = len(b_low_motion)
        best_length = current_length

    if best_length == 0:
        return {
            "longest_low_motion_frames": 0,
            "longest_low_motion_duration_sec": 0.0,
            "start_timestamp_ms": None,
            "end_timestamp_ms": None,
            "mean_step_distance_m": float(delta_distance.mean()) if len(delta_distance) > 0 else 0.0,
        }

    start_timestamp_ms = int(timestamp_array[best_start_idx])
    end_timestamp_ms = int(timestamp_array[best_end_idx])
    longest_low_motion_duration_sec = (end_timestamp_ms - start_timestamp_ms) / 1000.0

    return {
        "longest_low_motion_frames": best_length + 1,
        "longest_low_motion_duration_sec": longest_low_motion_duration_sec,
        "start_timestamp_ms": start_timestamp_ms,
        "end_timestamp_ms": end_timestamp_ms,
        "mean_step_distance_m": float(delta_distance.mean()) if len(delta_distance) > 0 else 0.0,
    }


def FindPoseFile(sequence_dir: str):
    output_dir = os.path.join(sequence_dir, "output")
    candidate_file_list = [
        os.path.join(output_dir, "pose_slam_euler_hba_refined.txt"),
        os.path.join(output_dir, "pose_slam_euler.txt"),
        os.path.join(output_dir, "pose_slam_euler_waymo.txt"),
    ]

    for pose_file_path in candidate_file_list:
        if os.path.exists(pose_file_path):
            return pose_file_path

    return None


def ScanWaymoForMapSeg(
    root_dir: str,
    translation_thresh_meters: float = 0.03,
    min_duration_sec: float = 2.0,
    top_k: int = 30,
):
    result_list = []

    sequence_name_list = sorted(
        file_name
        for file_name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, file_name))
    )

    for sequence_name in sequence_name_list:
        sequence_dir = os.path.join(root_dir, sequence_name)
        pose_file_path = FindPoseFile(sequence_dir)
        if pose_file_path is None:
            continue

        timestamp_array, pose_array = LoadPoseFile(pose_file_path)
        if len(timestamp_array) < 2:
            continue

        sequence_result = FindLongestLowMotionSegment(
            timestamp_array=timestamp_array,
            pose_array=pose_array,
            translation_thresh_meters=translation_thresh_meters,
        )
        sequence_result["sequence_name"] = sequence_name
        sequence_result["pose_file_path"] = pose_file_path
        result_list.append(sequence_result)

    result_list.sort(
        key=lambda item: item["longest_low_motion_duration_sec"],
        reverse=True
    )

    print(f"root_dir = {root_dir}")
    print(f"translation_thresh_meters = {translation_thresh_meters}")
    print(f"min_duration_sec = {min_duration_sec}")
    print(f"num_sequences = {len(result_list)}")
    print("")
    print("## 长时间不怎么动的序列")

    rank_idx = 0
    for result in result_list:
        if result["longest_low_motion_duration_sec"] < min_duration_sec:
            continue

        rank_idx += 1
        print(
            f"{rank_idx:02d}. {result['sequence_name']} | "
            f"longest_low_motion_duration_sec={result['longest_low_motion_duration_sec']:.3f} | "
            f"longest_low_motion_frames={result['longest_low_motion_frames']} | "
            f"start={result['start_timestamp_ms']} | "
            f"end={result['end_timestamp_ms']} | "
            f"mean_step_distance_m={result['mean_step_distance_m']:.4f}"
        )
        if rank_idx >= top_k:
            break


def Main():
    ScanWaymoForMapSeg(
        root_dir=os.environ["WAYMO_DATA_ROOT"],
        translation_thresh_meters=0.03,
        min_duration_sec=2.0,
        top_k=30,
    )


if __name__ == "__main__":
    Main()


"""
## 长时间不怎么动的序列
01. 5144634012371033641_920_000_940_000 | longest_low_motion_duration_sec=19.800 | longest_low_motion_frames=199 | start=1554243497822 | end=1554243517622 | mean_step_distance_m=0.0002
02. 8327447186504415549_5200_000_5220_000 | longest_low_motion_duration_sec=19.800 | longest_low_motion_frames=199 | start=1543276148562 | end=1543276168362 | mean_step_distance_m=0.0002
03. 15696964848687303249_4615_200_4635_200 | longest_low_motion_duration_sec=19.701 | longest_low_motion_frames=198 | start=1507241617619 | end=1507241637320 | mean_step_distance_m=0.0002
04. 6606076833441976341_1340_000_1360_000 | longest_low_motion_duration_sec=19.700 | longest_low_motion_frames=198 | start=1547744136737 | end=1547744156437 | mean_step_distance_m=0.0001
05. 4457475194088194008_3100_000_3120_000 | longest_low_motion_duration_sec=18.000 | longest_low_motion_frames=181 | start=1549651357289 | end=1549651375289 | mean_step_distance_m=0.0077
06. 14133920963894906769_1480_000_1500_000 | longest_low_motion_duration_sec=17.900 | longest_low_motion_frames=180 | start=1558033977547 | end=1558033995447 | mean_step_distance_m=0.0103
07. 18295766828140813622_6775_000_6795_000 | longest_low_motion_duration_sec=17.801 | longest_low_motion_frames=179 | start=1520779920580 | end=1520779938381 | mean_step_distance_m=0.0101
08. 12505030131868863688_1740_000_1760_000 | longest_low_motion_duration_sec=17.200 | longest_low_motion_frames=173 | start=1558483877522 | end=1558483894722 | mean_step_distance_m=0.0321
09. 204421859195625800_1080_000_1100_000 | longest_low_motion_duration_sec=16.900 | longest_low_motion_frames=170 | start=1557159634922 | end=1557159651822 | mean_step_distance_m=0.0300
10. 14098575914252965554_511_000_531_000 | longest_low_motion_duration_sec=15.300 | longest_low_motion_frames=146 | start=1521563062601 | end=1521563077901 | mean_step_distance_m=0.0836
11. 9521653920958139982_940_000_960_000 | longest_low_motion_duration_sec=15.300 | longest_low_motion_frames=154 | start=1557886617047 | end=1557886632347 | mean_step_distance_m=0.0146
12. 10023947602400723454_1120_000_1140_000 | longest_low_motion_duration_sec=14.800 | longest_low_motion_frames=149 | start=1552440200362 | end=1552440215162 | mean_step_distance_m=0.0382
13. 15550613280008674010_1780_000_1800_000 | longest_low_motion_duration_sec=12.400 | longest_low_motion_frames=125 | start=1552948830131 | end=1552948842531 | mean_step_distance_m=0.1656
14. 11113047206980595400_2560_000_2580_000 | longest_low_motion_duration_sec=10.900 | longest_low_motion_frames=110 | start=1557855055247 | end=1557855066147 | mean_step_distance_m=0.1286
15. 16578409328451172992_3780_000_3800_000 | longest_low_motion_duration_sec=9.200 | longest_low_motion_frames=93 | start=1557341343122 | end=1557341352322 | mean_step_distance_m=0.2625
16. 6771922013310347577_4249_290_4269_290 | longest_low_motion_duration_sec=7.800 | longest_low_motion_frames=79 | start=1511659165386 | end=1511659173186 | mean_step_distance_m=0.1295
17. 12339284075576056695_1920_000_1940_000 | longest_low_motion_duration_sec=7.601 | longest_low_motion_frames=77 | start=1557335962548 | end=1557335970149 | mean_step_distance_m=0.1579
18. 16087604685956889409_40_000_60_000 | longest_low_motion_duration_sec=7.000 | longest_low_motion_frames=71 | start=1553206988774 | end=1553206995774 | mean_step_distance_m=0.1705
19. 15844593126368860820_3260_000_3280_000 | longest_low_motion_duration_sec=6.600 | longest_low_motion_frames=67 | start=1559178767237 | end=1559178773837 | mean_step_distance_m=0.0534
20. 17818548625922145895_1372_430_1392_430 | longest_low_motion_duration_sec=6.500 | longest_low_motion_frames=66 | start=1513637703532 | end=1513637710032 | mean_step_distance_m=0.5016
21. 4967385055468388261_720_000_740_000 | longest_low_motion_duration_sec=5.700 | longest_low_motion_frames=58 | start=1552658071347 | end=1552658077047 | mean_step_distance_m=0.2972
22. 15535062863944567958_1100_000_1120_000 | longest_low_motion_duration_sec=3.900 | longest_low_motion_frames=40 | start=1557844308522 | end=1557844312422 | mean_step_distance_m=0.6128
23. 3644145307034257093_3000_400_3020_400 | longest_low_motion_duration_sec=3.800 | longest_low_motion_frames=37 | start=1511657365258 | end=1511657369058 | mean_step_distance_m=0.7313
24. 1918764220984209654_5680_000_5700_000 | longest_low_motion_duration_sec=3.700 | longest_low_motion_frames=38 | start=1541710113724 | end=1541710117424 | mean_step_distance_m=0.2887
25. 4733704239941053266_960_000_980_000 | longest_low_motion_duration_sec=3.100 | longest_low_motion_frames=32 | start=1559176462379 | end=1559176465479 | mean_step_distance_m=0.3547
26. 18141076662151909970_2755_710_2775_710 | longest_low_motion_duration_sec=2.800 | longest_low_motion_frames=29 | start=1515100329332 | end=1515100332132 | mean_step_distance_m=0.6958
27. 2618605158242502527_1860_000_1880_000 | longest_low_motion_duration_sec=2.800 | longest_low_motion_frames=29 | start=1552675802277 | end=1552675805077 | mean_step_distance_m=0.2999
28. 14106113060128637865_1200_000_1220_000 | longest_low_motion_duration_sec=2.600 | longest_low_motion_frames=27 | start=1547700372072 | end=1547700374672 | mean_step_distance_m=0.4922
29. 18418533015310829002_480_000_500_000 | longest_low_motion_duration_sec=2.401 | longest_low_motion_frames=25 | start=1550088116552 | end=1550088118953 | mean_step_distance_m=0.2707
30. 11967272535264406807_580_000_600_000 | longest_low_motion_duration_sec=2.400 | longest_low_motion_frames=25 | start=1557873610298 | end=1557873612698 | mean_step_distance_m=0.1667
"""


"""
5144634012371033641_920_000_940_000
4457475194088194008_3100_000_3120_000
14098575914252965554_511_000_531_000
10023947602400723454_1120_000_1140_000
15550613280008674010_1780_000_1800_000
11113047206980595400_2560_000_2580_000
6771922013310347577_4249_290_4269_290
4967385055468388261_720_000_740_000
15535062863944567958_1100_000_1120_000
11967272535264406807_580_000_600_000
16102220208346880_1420_000_1440_000
"""
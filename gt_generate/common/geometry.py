import numpy as np


def YprToRotationMatrixV2(yaw_pitch_roll):
    yaw = float(yaw_pitch_roll[0])
    pitch = float(yaw_pitch_roll[1])
    roll = float(yaw_pitch_roll[2])

    ci = np.cos(roll)
    cj = np.cos(pitch)
    ch = np.cos(yaw)
    si = np.sin(roll)
    sj = np.sin(pitch)
    sh = np.sin(yaw)
    cc = ci * ch
    cs = ci * sh
    sc = si * ch
    ss = si * sh

    return np.array(
        [
            [cj * ch, sj * sc - cs, sj * cc + ss],
            [cj * sh, sj * ss + cc, sj * cs - sc],
            [-sj, cj * si, cj * ci],
        ],
        dtype=np.float64,
    )



import numpy as np
import math




def YPR2RotationMatrixV2(ypr):
# R = Rot_z(yaw) * Rot_y(pitch) * Rot_x(roll)

    eulerZ = ypr[0]
    eulerY = ypr[1]
    eulerX = ypr[2]

    ci = math.cos(eulerX);
    cj = math.cos(eulerY);
    ch = math.cos(eulerZ);
    si = math.sin(eulerX);
    sj = math.sin(eulerY);
    sh = math.sin(eulerZ);

    cc = ci * ch;
    cs = ci * sh;
    sc = si * ch;
    ss = si * sh;
    
    R = np.array([[cj * ch, sj * sc - cs, sj * cc + ss],
                [cj * sh, sj * ss + cc, sj * cs - sc],
                [-sj, cj * si, cj * ci]])

    return R


def RotationMatrix2YPRV2(R):
    if 0:
        pitch = -math.asin(R[2,0])
        cos_pitch_abs = abs(math.cos(pitch))
        azimuth = math.atan2(R[1,0] / (cos_pitch_abs + 0.000001), R[0,0] / (cos_pitch_abs + 0.000001))
        roll = math.atan2(R[2,1] / (cos_pitch_abs + 0.000001), R[2,2] / (cos_pitch_abs + 0.000001))
        return np.array([roll, pitch, azimuth])
    else:
        EPS = 1e-6

        # 限制范围避免asin报错
        r20 = max(-1.0, min(1.0, R[2, 0]))
        pitch = -math.asin(r20)
        cos_pitch = math.cos(pitch)

        if abs(cos_pitch) < EPS:  # pitch接近±90度
            yaw = math.atan2(-R[0, 1], R[1, 1])
            roll = 0.0
        else:
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])

        return np.array([yaw, pitch, roll])
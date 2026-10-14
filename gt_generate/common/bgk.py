import numpy as np
from scipy import ndimage

from .constants import INVALID_HEIGHT, PI


class BgkSmoother:
    def __init__(self, map_resolution, prediction_kernel_radius, valid_points_num_thresh, bilateral_std):
        self.map_resolution = float(map_resolution)
        self.prediction_kernel_radius = float(prediction_kernel_radius)
        self.valid_points_num_thresh = int(valid_points_num_thresh)
        self.bilateral_std = float(bilateral_std)
        self.kernel, self.count_kernel, self.offsets = self._BuildKernel()
        self.max_conf = float(np.sum(self.kernel))
        self.lut = np.exp(-((np.arange(200, dtype=np.float64) * 0.01) ** 2) / (2.0 * self.bilateral_std * self.bilateral_std))

    def _BuildKernel(self):
        grid_radius = int(self.prediction_kernel_radius / self.map_resolution)
        kernel = np.zeros((2 * grid_radius + 1, 2 * grid_radius + 1), dtype=np.float64)
        count_kernel = np.zeros_like(kernel, dtype=np.float64)
        offsets = []
        for row_idx in range(kernel.shape[0]):
            for col_idx in range(kernel.shape[1]):
                dist = np.hypot(row_idx - grid_radius, col_idx - grid_radius) * self.map_resolution
                if dist > self.prediction_kernel_radius:
                    continue
                d_value = dist / (self.prediction_kernel_radius + 0.01)
                weight = (2.0 + np.cos(2.0 * PI * d_value)) / 3.0 * (1.0 - d_value) + np.sin(2.0 * PI * d_value) / (2.0 * PI)
                kernel[row_idx, col_idx] = weight
                count_kernel[row_idx, col_idx] = 1.0
                offsets.append((row_idx - grid_radius, col_idx - grid_radius, weight))
        return kernel, count_kernel, offsets

    def Run(self, input_map, invalid_value=INVALID_HEIGHT, b_perform_bf=True):
        valid_mask = np.isfinite(input_map) & (input_map != invalid_value)
        values = np.where(valid_mask, input_map, 0.0)

        sum_ky = ndimage.convolve(values, self.kernel, mode="constant", cval=0.0)
        sum_k = ndimage.convolve(valid_mask.astype(np.float64), self.kernel, mode="constant", cval=0.0)
        valid_point_num = ndimage.convolve(valid_mask.astype(np.float64), self.count_kernel, mode="constant", cval=0.0)

        result_map = np.full(input_map.shape, invalid_value, dtype=np.float64)
        pred_mask = (sum_k > 0.0) & (valid_point_num > self.valid_points_num_thresh)
        result_map[pred_mask] = sum_ky[pred_mask] / sum_k[pred_mask]

        if not b_perform_bf:
            conf_map = np.zeros(input_map.shape, dtype=np.float64)
            conf_map[pred_mask] = sum_k[pred_mask]
            return result_map, conf_map

        bf_sum_ky = np.zeros(input_map.shape, dtype=np.float64)
        bf_sum_k = np.zeros(input_map.shape, dtype=np.float64)
        bf_count = np.zeros(input_map.shape, dtype=np.float64)
        refined_valid_mask = pred_mask

        for row_offset, col_offset, weight in self.offsets:
            shifted_input = np.roll(values, shift=(row_offset, col_offset), axis=(0, 1))
            shifted_mask = np.roll(valid_mask, shift=(row_offset, col_offset), axis=(0, 1))
            if row_offset > 0:
                shifted_mask[:row_offset, :] = False
            elif row_offset < 0:
                shifted_mask[row_offset:, :] = False
            if col_offset > 0:
                shifted_mask[:, :col_offset] = False
            elif col_offset < 0:
                shifted_mask[:, col_offset:] = False

            use_mask = shifted_mask & refined_valid_mask
            if not np.any(use_mask):
                continue
            diff_map = np.abs(shifted_input - result_map)
            lut_idx = np.clip((diff_map * 100.0).astype(np.int32), 0, 199)
            weights = self.lut[lut_idx] * weight
            bf_sum_ky[use_mask] += weights[use_mask] * shifted_input[use_mask]
            bf_sum_k[use_mask] += weights[use_mask]
            bf_count[use_mask] += 1.0

        final_map = np.full(input_map.shape, invalid_value, dtype=np.float64)
        final_mask = (bf_sum_k > 0.0) & (bf_count > self.valid_points_num_thresh)
        final_map[final_mask] = bf_sum_ky[final_mask] / bf_sum_k[final_mask]

        conf_map = np.zeros(input_map.shape, dtype=np.float64)
        conf_map[final_mask] = bf_sum_k[final_mask]
        return final_map, conf_map


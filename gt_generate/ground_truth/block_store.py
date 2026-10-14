import os
from dataclasses import dataclass

import numpy as np

from gt_generate.common.io_utils import EnsureDir


@dataclass
class TerrainBlock:
    block_x: int
    block_y: int
    rows: int
    cols: int
    count: np.ndarray
    sum_z: np.ndarray
    sum_z2: np.ndarray
    road_count: np.ndarray
    road2_count: np.ndarray


class TerrainBlockStore:
    def __init__(self, output_dir, map_resolution_m=0.2, block_rows=200):
        self.output_dir = str(output_dir)
        self.block_dir = os.path.join(self.output_dir, "blocks")
        self.map_resolution_m = float(map_resolution_m)
        self.block_rows = int(block_rows)
        self.blocks = {}
        EnsureDir(self.block_dir)

    def GetBlockKeyFromCell(self, cell_x, cell_y):
        block_x = int(np.floor(float(cell_x) / float(self.block_rows)))
        block_y = int(np.floor(float(cell_y) / float(self.block_rows)))
        return block_x, block_y

    def GetOrCreateBlock(self, block_key):
        if block_key in self.blocks:
            return self.blocks[block_key]
        block = TerrainBlock(
            block_x=int(block_key[0]),
            block_y=int(block_key[1]),
            rows=self.block_rows,
            cols=self.block_rows,
            count=np.zeros((self.block_rows, self.block_rows), dtype=np.uint32),
            sum_z=np.zeros((self.block_rows, self.block_rows), dtype=np.float64),
            sum_z2=np.zeros((self.block_rows, self.block_rows), dtype=np.float64),
            road_count=np.zeros((self.block_rows, self.block_rows), dtype=np.uint32),
            road2_count=np.zeros((self.block_rows, self.block_rows), dtype=np.uint32),
        )
        self.blocks[block_key] = block
        return block

    def AddWorldPoints(self, points_xyz, ground_type=None):
        if points_xyz.shape[0] == 0:
            return
        if ground_type is None:
            ground_type = np.zeros((points_xyz.shape[0],), dtype=np.uint8)
        ground_type = np.asarray(ground_type, dtype=np.uint8).reshape(-1)
        if ground_type.shape[0] != points_xyz.shape[0]:
            raise ValueError(f"ground_type size mismatch: {ground_type.shape[0]} vs {points_xyz.shape[0]}")

        cell_x = np.floor(points_xyz[:, 0] / self.map_resolution_m).astype(np.int64)
        cell_y = np.floor(points_xyz[:, 1] / self.map_resolution_m).astype(np.int64)
        block_x = np.floor_divide(cell_x, self.block_rows)
        block_y = np.floor_divide(cell_y, self.block_rows)
        local_x = np.mod(cell_x, self.block_rows).astype(np.int32)
        local_y = np.mod(cell_y, self.block_rows).astype(np.int32)
        z_values = points_xyz[:, 2].astype(np.float64)

        block_keys = np.unique(np.column_stack((block_x, block_y)), axis=0)
        for block_key_array in block_keys:
            bx = int(block_key_array[0])
            by = int(block_key_array[1])
            select_mask = (block_x == bx) & (block_y == by)
            if not np.any(select_mask):
                continue
            block = self.GetOrCreateBlock((bx, by))
            rows = self.block_rows - 1 - local_y[select_mask]
            cols = local_x[select_mask]
            current_z = z_values[select_mask]
            np.add.at(block.count, (rows, cols), 1)
            np.add.at(block.sum_z, (rows, cols), current_z)
            np.add.at(block.sum_z2, (rows, cols), current_z * current_z)
            current_ground_type = ground_type[select_mask]
            np.add.at(block.road_count, (rows[current_ground_type == 1], cols[current_ground_type == 1]), 1)
            np.add.at(block.road2_count, (rows[current_ground_type == 2], cols[current_ground_type == 2]), 1)

    def SaveBlocks(self):
        for block in self.blocks.values():
            filename = os.path.join(self.block_dir, f"block_{block.block_x}_{block.block_y}.npz")
            np.savez_compressed(
                filename,
                block_x=np.int32(block.block_x),
                block_y=np.int32(block.block_y),
                count=block.count,
                sum_z=block.sum_z,
                sum_z2=block.sum_z2,
                road_count=block.road_count,
                road2_count=block.road2_count,
            )

    def BuildDenseStats(self):
        if not self.blocks:
            raise RuntimeError("No terrain blocks were accumulated.")
        block_x_values = [key[0] for key in self.blocks.keys()]
        block_y_values = [key[1] for key in self.blocks.keys()]
        min_block_x = int(np.min(block_x_values))
        max_block_x = int(np.max(block_x_values))
        min_block_y = int(np.min(block_y_values))
        max_block_y = int(np.max(block_y_values))
        rows = (max_block_y - min_block_y + 1) * self.block_rows
        cols = (max_block_x - min_block_x + 1) * self.block_rows
        count_map = np.zeros((rows, cols), dtype=np.uint32)
        sum_z_map = np.zeros((rows, cols), dtype=np.float64)
        sum_z2_map = np.zeros((rows, cols), dtype=np.float64)
        road_count_map = np.zeros((rows, cols), dtype=np.uint32)
        road2_count_map = np.zeros((rows, cols), dtype=np.uint32)

        for (block_x, block_y), block in self.blocks.items():
            row0 = (max_block_y - block_y) * self.block_rows
            col0 = (block_x - min_block_x) * self.block_rows
            count_map[row0 : row0 + self.block_rows, col0 : col0 + self.block_rows] = block.count
            sum_z_map[row0 : row0 + self.block_rows, col0 : col0 + self.block_rows] = block.sum_z
            sum_z2_map[row0 : row0 + self.block_rows, col0 : col0 + self.block_rows] = block.sum_z2
            road_count_map[row0 : row0 + self.block_rows, col0 : col0 + self.block_rows] = block.road_count
            road2_count_map[row0 : row0 + self.block_rows, col0 : col0 + self.block_rows] = block.road2_count

        map_min_x_cm = int(round(min_block_x * self.block_rows * self.map_resolution_m * 100.0))
        map_min_y_cm = int(round(min_block_y * self.block_rows * self.map_resolution_m * 100.0))
        return {
            "map_min_x_cm": map_min_x_cm,
            "map_min_y_cm": map_min_y_cm,
            "rows": rows,
            "cols": cols,
            "count_map": count_map,
            "sum_z_map": sum_z_map,
            "sum_z2_map": sum_z2_map,
            "road_count_map": road_count_map,
            "road2_count_map": road2_count_map,
        }

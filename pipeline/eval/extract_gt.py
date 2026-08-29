"""FARO Laser Point Cloud Ground Truth Extractor.

Parses binary and ASCII PLY point cloud headers dynamically to extract
vertex coordinates (x, y, z), point count loaded, floor plane z,
ceiling plane z, and reference room height via SVD horizontal plane fitting.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]

PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


class IncompleteDownloadError(ValueError):
    """Raised when a PLY file download is truncated or incomplete."""
    pass


def read_ply_points(ply_path: Path, strict_completeness: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Parses binary or ASCII PLY header dynamically and returns (N, 3) vertex array + header metadata."""
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY point cloud file not found: {ply_path}")

    with open(ply_path, "rb") as f:
        header_lines = []
        format_str = ""
        vertex_count = 0
        properties = []
        current_element = None
        header_bytes = 0

        while True:
            line_bytes = f.readline()
            header_bytes += len(line_bytes)
            line = line_bytes.decode("latin-1").strip()
            header_lines.append(line)
            if line.startswith("format"):
                format_str = line
            elif line.startswith("element"):
                parts = line.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif line.startswith("property") and current_element == "vertex":
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"PLY list properties unsupported in vertex element: {line}")
                properties.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        if not format_str:
            raise ValueError(f"Invalid PLY header in {ply_path}: missing format statement")
        if vertex_count == 0:
            raise ValueError(f"Invalid PLY header in {ply_path}: 0 vertices declared")

        prop_names = [p[0] for p in properties]
        for coord in ("x", "y", "z"):
            if coord not in prop_names:
                raise ValueError(f"PLY header missing mandatory coordinate '{coord}'. Properties found: {prop_names}")

        metadata = {
            "format": format_str,
            "vertexCountDeclared": vertex_count,
            "headerBytes": header_bytes,
            "properties": properties,
        }

        if "format ascii" in format_str:
            data = np.loadtxt(f, max_rows=vertex_count)
            col_x = prop_names.index("x")
            col_y = prop_names.index("y")
            col_z = prop_names.index("z")
            return data[:, [col_x, col_y, col_z]], metadata

        if "format binary_little_endian" in format_str or "format binary_big_endian" in format_str:
            endian = "<" if "binary_little_endian" in format_str else ">"
            dtype_fields = []
            for p_name, p_type in properties:
                if p_type not in PLY_TYPE_MAP:
                    raise ValueError(f"Unsupported PLY property data type '{p_type}' for '{p_name}'")
                dtype_fields.append((p_name, endian + PLY_TYPE_MAP[p_type]))

            dtype = np.dtype(dtype_fields)
            raw_payload = f.read()
            expected_bytes = vertex_count * dtype.itemsize

            if strict_completeness and len(raw_payload) < expected_bytes:
                raise IncompleteDownloadError(
                    f"File download incomplete for {ply_path.name}: expected {expected_bytes:,} bytes, "
                    f"got {len(raw_payload):,} bytes ({len(raw_payload)/expected_bytes:.1%} complete)."
                )

            available_vertices = len(raw_payload) // dtype.itemsize
            vertices_to_read = min(vertex_count, available_vertices)

            if vertices_to_read == 0:
                raise ValueError(f"No vertex payload data available in PLY file {ply_path}")

            data = np.frombuffer(raw_payload[:vertices_to_read * dtype.itemsize], dtype=dtype)
            x = np.nan_to_num(np.asarray(data["x"], dtype=np.float64), nan=0.0)
            y = np.nan_to_num(np.asarray(data["y"], dtype=np.float64), nan=0.0)
            z = np.nan_to_num(np.asarray(data["z"], dtype=np.float64), nan=0.0)
            xyz = np.column_stack([x, y, z])
            valid_mask = (x != 0.0) | (y != 0.0) | (z != 0.0)
            xyz = xyz[valid_mask]
            metadata["verticesLoaded"] = len(xyz)
            return xyz, metadata

        raise ValueError(f"Unsupported PLY format: {format_str}")


def extract_ground_truth_from_ply(
    ply_path: Path | str,
    pose_path: Path | str | None = None,
    strict_completeness: bool = True
) -> dict[str, Any]:
    """Parses a PLY point cloud file dynamically, applies pose matrix T if present, and fits SVD floor/ceiling planes."""
    path = Path(ply_path)
    points, metadata = read_ply_points(path, strict_completeness=strict_completeness)

    # Apply 4x4 laser pose matrix if provided
    if pose_path and Path(pose_path).is_file():
        pose_lines = [line.strip() for line in Path(pose_path).read_text().strip().split("\n") if line.strip()]
        pose_rows = [[float(x) for x in line.split(",")] for line in pose_lines]
        T = np.array(pose_rows, dtype=np.float64)
        R = T[:3, :3]
        t = T[3, :3] if T[3, 3] == 1.0 and abs(T[3, 2]) > 1.0 else T[:3, 3]
        points = points @ R.T + t

    z_coords = points[:, 2]

    # SVD Horizontal plane extraction
    counts, bin_edges = np.histogram(z_coords, bins=max(1, int(np.ceil((z_coords.max() - z_coords.min()) / 0.02))))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    fractions = counts / counts.sum()

    supported = np.nonzero(fractions >= 0.015)[0]
    groups: list[list[int]] = []
    for index in supported:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    bands = [(float(np.average(bin_centers[g], weights=counts[g])), float(fractions[g].sum())) for g in groups]
    pairs = [(low, high, lw + hw) for i, (low, lw) in enumerate(bands) for (high, hw) in bands[i + 1:] if 2.0 <= (high - low) <= 4.0]

    if pairs:
        floor_c, ceiling_c, _ = max(pairs, key=lambda item: item[2])
        floor_band = points[np.abs(points[:, 2] - floor_c) <= 0.06]
        ceil_band = points[np.abs(points[:, 2] - ceiling_c) <= 0.06]

        if len(floor_band) >= 50 and len(ceil_band) >= 50:
            floor_centroid = floor_band.mean(axis=0)
            ceil_centroid = ceil_band.mean(axis=0)
            _, _, vt_f = np.linalg.svd(floor_band - floor_centroid, full_matrices=False)
            _, _, vt_c = np.linalg.svd(ceil_band - ceil_centroid, full_matrices=False)
            norm_f, norm_c = vt_f[-1], vt_c[-1]
            if norm_f[2] < 0: norm_f = -norm_f
            if norm_c[2] < 0: norm_c = -norm_c
            normal = (norm_f + norm_c) / 2.0
            normal = normal / np.linalg.norm(normal)
            height_m = round(abs(float((ceil_centroid - floor_centroid) @ normal)), 4)
            floor_z, ceiling_z = float(floor_centroid[2]), float(ceil_centroid[2])
        else:
            floor_z, ceiling_z = floor_c, ceiling_c
            height_m = round(ceiling_z - floor_z, 4)
    else:
        floor_z = float(np.percentile(z_coords, 1.0))
        ceiling_z = float(np.percentile(z_coords, 99.0))
        height_m = round(ceiling_z - floor_z, 4)

    return {
        "plyPath": str(path),
        "fileSizeBytes": path.stat().st_size if path.exists() else 0,
        "format": metadata["format"],
        "propertiesParsed": metadata["properties"],
        "pointCountLoaded": len(points),
        "floorPlaneZ": round(floor_z, 4),
        "ceilingPlaneZ": round(ceiling_z, 4),
        "extractedHeight_m": height_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reference ground truth height from a PLY laser point cloud file.")
    parser.add_argument("--ply", required=True, help="Path to raw .ply laser point cloud file.")
    parser.add_argument("--pose", help="Path to 4x4 _pose.txt pose matrix file.")
    parser.add_argument("--strict", action="store_true", default=True, help="Fail if PLY file download is incomplete.")
    args = parser.parse_args()

    res = extract_ground_truth_from_ply(args.ply, pose_path=args.pose, strict_completeness=args.strict)
    print("=== REAL FARO LASER POINT CLOUD SVD EXTRACTION ===")
    print(f"File Path: {res['plyPath']} ({res['fileSizeBytes']:,} bytes on disk)")
    print(f"Format: {res['format']}")
    print(f"Properties Parsed: {res['propertiesParsed']}")
    print(f"Point Count Loaded: {res['pointCountLoaded']:,}")
    print(f"Floor Plane Z: {res['floorPlaneZ']} m")
    print(f"Ceiling Plane Z: {res['ceilingPlaneZ']} m")
    print(f"Extracted Storey Height: {res['extractedHeight_m']} m")


if __name__ == "__main__":
    main()

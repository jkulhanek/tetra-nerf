import json
import shutil
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from ..utils import colmap_utils
from .utils import CONSOLE, parse_colmap_camera_params, run_command, status
from .utils import transform_poses as _transform_poses


def colmap_to_json(
    input_path: Path,
    output: Path,
):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file."""

    cameras = colmap_utils.read_cameras_binary(input_path / "cameras.bin")
    images = colmap_utils.read_images_binary(input_path / "images.bin")

    frames = []
    for im_data in images.values():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        file_path = Path(f"./images/{im_data.name}")
        frame = {
            "file_path": file_path.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    out = parse_colmap_camera_params(cameras[1])
    # mitetranerf360 uses this
    # But we dont have to!
    # out["orientation_override"] = "pca"
    out["frames"] = frames

    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


# pylint: disable=redefined-builtin
def colmap_to_minimal_parser_format(
    input: Path,
    output: Path,
    downscale_factor: int = 4,
    transform_poses: bool = True,
):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file."""

    cameras = colmap_utils.read_cameras_binary(input / "sparse" / "0" / "cameras.bin")
    images = colmap_utils.read_images_binary(input / "sparse" / "0" / "images.bin")
    images = list(images.values())
    images.sort(key=lambda x: x.name)

    c2ws = []
    file_paths = []
    for i, im_data in enumerate(images):
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2ws.append(c2w)
        file_path = Path(f"images_{downscale_factor}/{im_data.name}")
        file_paths.append(str(file_path))
        if i == 0:
            real_width, real_height = Image.open(input / file_path).size
    c2ws = np.stack(c2ws, 0)
    out = {}
    if transform_poses:
        c2ws, applied_transform, applied_scale = _transform_poses(c2ws)
        out["applied_transform"] = applied_transform
        out["applied_scale"] = applied_scale

    cam_par = parse_colmap_camera_params(cameras[1])
    cameras = {}
    if "k1" in cam_par:
        cameras["distortion_params"] = np.array(
            (
                cam_par.get("k1", 0.0),
                cam_par.get("k2", 0.0),
                cam_par.get("k3", 0.0),
                cam_par.get("k4", 0.0),
                cam_par.get("p1", 0.0),
                cam_par.get("p2", 0.0),
            ),
            dtype=np.float32,
        )
    scale_factor = real_width / cam_par["w"]
    dtype = np.float32
    cameras["fx"] = np.array(cam_par["fl_x"] * scale_factor, dtype)
    cameras["fy"] = np.array(cam_par["fl_y"] * scale_factor, dtype)
    cameras["cx"] = np.array(cam_par["cx"] * scale_factor, dtype)
    cameras["cy"] = np.array(cam_par["cy"] * scale_factor, dtype)
    cameras["width"] = np.array(real_width, np.int32)
    cameras["height"] = np.array(real_height, np.int32)
    cameras["camera_type"] = np.array(1, np.int32)
    out["scene_box"] = np.array([[-1, -1, -1], [1, 1, 1]], dtype=dtype)

    eval_interval = 8
    all_indices = np.arange(len(file_paths))
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    for split in ["train", "val", "test"]:
        # Export poses for MinimalDataLoader
        indices = train_indices if split == "train" else eval_indices
        out_fname = str(output / f"{split}.npz")
        np.savez(
            out_fname,
            image_filenames=[file_paths[i] for i in indices],
            cameras=np.array(dict(camera_to_worlds=c2ws[indices].astype(dtype), **cameras)),
            **out,
        )
        CONSOLE.log(f"Split {split} saved to {out_fname}")


def entrypoint(path: Path, verbose: bool = True, run_dense: bool = True, downscale_factor: int = 4):
    if not (path / "sparse" / "0").exists():
        CONSOLE.print("[red bold]The folder 'sparse/0' does not exist in the input path.")
    if not (path / "images").exists():
        CONSOLE.print("[red bold]The folder 'images' does not exist in the input path.")
    if not (path / "images").exists():
        CONSOLE.print("[red bold]The folder 'images_{{downscale_factor}}' does not exist in the input path.")

    # colmap_to_json(path/"sparse"/"0", path/"transforms.json")
    # CONSOLE.print(f"[green bold]File transforms.json generated in [yellow]{path/'transforms.json'}")
    colmap_to_minimal_parser_format(path, path, downscale_factor)
    CONSOLE.print(f"[green bold]Poses for minimal-parser generated to [yellow]{path}")

    # Generate sparse model
    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            path / "sparse" / "0",
            "--output_path",
            path / "sparse.ply",
            "--output_type",
            "PLY",
        ],
        verbose=verbose,
    )
    CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{path/'sparse.ply'}")

    if run_dense:
        with status(status="[bold yellow]Generating dense model...", visible=not verbose):
            run_command(
                [
                    "colmap",
                    "image_undistorter",
                    "--input_path",
                    path / "sparse" / "0",
                    "--output_path",
                    path / "dense",
                    "--image_path",
                    path / "images",
                ],
                verbose=verbose,
            )

            run_command(
                ["colmap", "patch_match_stereo", "--workspace_path", path / "dense"],
                verbose=verbose,
            )

            run_command(
                [
                    "colmap",
                    "stereo_fusion",
                    "--output_path",
                    path / "dense" / "fused.ply",
                    "--workspace_path",
                    path / "dense",
                ],
                verbose=verbose,
            )

            shutil.copy(path / "dense/fused.ply", path / "dense.ply")
            CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{path/'dense.ply'}")


if __name__ == "__main__":
    tyro.cli(entrypoint)

import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from PIL import Image

from ..utils import colmap_utils
from .utils import CONSOLE, parse_colmap_camera_params, run_command, status
from .utils import transform_poses as _transform_poses

MAX_AUTO_RESOLUTION = 1600


# pylint: disable=redefined-builtin
def colmap_to_minimal_parser_format(
    input: Path,
    output: Path,
    train_images,
    downscale_factor: int,
    transform_poses: bool = True,
):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file."""

    cameras = colmap_utils.read_cameras_binary(input / "cameras.bin")
    images = colmap_utils.read_images_binary(input / "images.bin")
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
            real_width, real_height = Image.open(output / file_path).size
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
        CONSOLE.print(f"Using distortion parameters: {cameras['distortion_params'].tolist()}")
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

    train_images_set = set(train_images)
    train_indices = [x for x in range(len(file_paths)) if Path(file_paths[x]).name in train_images_set]
    eval_indices = [x for x in range(len(file_paths)) if x not in train_indices]
    assert len(train_indices) > 0
    assert len(eval_indices) > 0
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


def get_default_downscale_factor(images_path: Path):
    test_img = Image.open(next(iter(images_path.glob("*"))))
    h, w = test_img.size
    max_res = max(h, w)
    df = 0
    while True:
        if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
            break
        df += 1
    downscale_factor = 2**df
    CONSOLE.log(f"Auto image downscale factor of {downscale_factor}")
    return downscale_factor


def downscale_images(source, target, downscale_factor, verbose):
    assert source != target
    with status("[bold yellow]Downscaling images...", visible=not verbose):
        assert isinstance(downscale_factor, int)
        target.mkdir(parents=True, exist_ok=True)
        for f in source.glob("*"):
            out = target / f.name
            if not out.exists():
                # Downscale and add white background (only for Ignatius which is corrupted in the NSVF download)
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-noautorotate",
                    "-i",
                    str(f),
                    "-q:v",
                    "2",
                    "-filter_complex",
                    f"[v:0]scale=iw/{downscale_factor}:ih/{downscale_factor}",
                    "-frames:v",
                    "1",
                    str(out),
                ]
                run_command(ffmpeg_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    CONSOLE.log(f"Images saved to {target}")


# pylint: disable=too-many-statements
def entrypoint(
    path: Path,
    camera_model: str = "SIMPLE_RADIAL",
    verbose: bool = True,
    exhaustive_matching: bool = True,
    run_sparse: bool = True,
    run_dense: bool = False,
    downscale_factor: Optional[int] = None,
    transform_poses: bool = True,
    separate_training_pointcloud: bool = False,
):
    if not (path / "images").exists():
        CONSOLE.print(f"The path {path} does not contain an images folder.")
        sys.exit(1)

    if downscale_factor is None:
        downscale_factor = get_default_downscale_factor(path / "images")

    # Build sparse model for the poses
    if not (path / "sparse/0").exists():
        with status(
            status="[bold yellow]Generating sparse model for poses...",
            visible=not verbose,
        ):
            (path / "sparse").mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(path / "sparse" / "database.db"),
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_model",
                    camera_model,
                    "--image_path",
                    str(path / "images"),
                ],
                verbose=verbose,
            )
            if exhaustive_matching:
                run_command(
                    [
                        "colmap",
                        "exhaustive_matcher",
                        "--database_path",
                        str(path / "sparse" / "database.db"),
                    ]
                )
            else:
                run_command(
                    [
                        "colmap",
                        "sequential_matcher",
                        # '--SequentialMatching.quadratic_overlap', '0',
                        "--database_path",
                        str(path / "sparse" / "database.db"),
                    ]
                )
            (path / "sparse").mkdir(exist_ok=True, parents=True)
            run_command(
                [
                    "colmap",
                    "mapper",
                    "--database_path",
                    path / "sparse" / "database.db",
                    "--image_path",
                    str(path / "images"),
                    "--output_path",
                    str(path / "sparse"),
                ]
            )

    scaled_images_path = path / "images"
    if downscale_factor > 1:
        scaled_images_path = path / f"images_{downscale_factor}"
    downscale_images(path / "images", scaled_images_path, downscale_factor, verbose)

    # For the actual pointclouds, we only use training images
    db = colmap_utils.COLMAPDatabase(path / "sparse" / "database.db")
    db_images = db.execute("select * from images").fetchall()
    id_map = {x[1]: x[0] for x in db_images}
    all_filepaths = sorted(list(id_map.keys()))
    eval_filepaths = all_filepaths[::8]
    train_filepaths = sorted(set(all_filepaths).difference(eval_filepaths))
    all_images = colmap_utils.read_images_binary(path / "sparse" / "0" / "images.bin")

    if not (path / "test.npz").exists():
        # Generate train val test data
        colmap_to_minimal_parser_format(
            path / "sparse/0",
            path,
            train_filepaths,
            downscale_factor,
            transform_poses=transform_poses,
        )

    if not (path / "sparse.ply").exists() and run_sparse:
        sparse_path = path / "sparse" / "0"
        if separate_training_pointcloud:
            sparse_path = path / "training_sparse"
            images = {}
            (path / "training_images").mkdir(exist_ok=True)
            for f in train_filepaths:
                if id_map[f] not in all_images:
                    continue  # Unmatched image
                images[id_map[f]] = all_images[id_map[f]]
                if not (path / "training_images" / f).exists():
                    shutil.copy(path / "images" / f, path / "training_images" / f)

            (path / "training_sparse").mkdir(exist_ok=True)
            shutil.copy(
                path / "sparse" / "0" / "cameras.bin",
                path / "training_sparse" / "cameras.bin",
            )
            shutil.copy(path / "sparse" / "database.db", path / "training_sparse" / "database.db")
            colmap_utils.write_images_binary(images, path / "training_sparse" / "images.bin")
            colmap_utils.write_points3D_binary({}, path / "training_sparse" / "points3D.bin")

            with status(status="[bold yellow]Generating sparse model...", visible=not verbose):
                run_command(
                    [
                        "colmap",
                        "point_triangulator",
                        "--clear_points",
                        "1",
                        "--database_path",
                        path / "training_sparse" / "database.db",
                        "--input_path",
                        path / "training_sparse",
                        "--output_path",
                        path / "training_sparse",
                        "--image_path",
                        path / "images",
                    ],
                    verbose=verbose,
                )

        run_command(
            [
                "colmap",
                "model_converter",
                "--input_path",
                sparse_path,
                "--output_path",
                sparse_path / "pointcloud.ply",
                "--output_type",
                "PLY",
            ],
            verbose=verbose,
        )
        shutil.copy(sparse_path / "pointcloud.ply", path / "sparse.ply")
        CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{path/'sparse.ply'}")

    # Get the dense model
    if run_dense and not (path / "dense.ply").exists():
        with status(status="[bold yellow]Generating dense model...", visible=not verbose):
            run_command(
                [
                    "colmap",
                    "image_undistorter",
                    "--input_path",
                    sparse_path,
                    "--output_path",
                    path / "training_dense",
                    "--image_path",
                    path / "training_images",
                ],
                verbose=verbose,
            )

            run_command(
                [
                    "colmap",
                    "patch_match_stereo",
                    "--workspace_path",
                    path / "training_dense",
                ],
                verbose=verbose,
            )

            run_command(
                [
                    "colmap",
                    "stereo_fusion",
                    "--output_path",
                    path / "training_dense" / "fused.ply",
                    "--workspace_path",
                    path / "training_dense",
                ],
                verbose=verbose,
            )

            shutil.copy(path / "training_dense/fused.ply", path / "dense.ply")
            CONSOLE.print(f"[green bold]Dense model stored in [yellow]{path/'dense.ply'}")


if __name__ == "__main__":
    tyro.cli(entrypoint)

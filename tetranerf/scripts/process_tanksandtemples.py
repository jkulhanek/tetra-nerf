import contextlib
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro
from PIL import Image
from rich.console import Console

from ..utils import colmap_utils
from .utils import transform_poses as _transform_poses

CONSOLE = Console(width=120)


def run_command(cmd, verbose=True) -> Optional[str]:
    """Runs a command and returns the output.
    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, check=False)
    if out.returncode != 0:
        CONSOLE.rule(
            "[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ",
            style="red",
        )
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        if out.stdout is not None:
            CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


def change_ply_coordinates_colmap_to_blender(pointcloud: Path, output: Path):
    pc = trimesh.load(pointcloud)
    vertices = pc.vertices
    vertices = vertices[:, np.array([1, 0, 2])]
    vertices[:, 2] *= -1
    pc.vertices = vertices
    pc.export(output)


def status(*args, spinner="bouncingBall", visible=True, **kwargs):
    if visible:
        return CONSOLE.status(*args, **kwargs, spinner=spinner, **kwargs)
    return contextlib.suppress()


# pylint: disable=too-many-statements
def entrypoint(
    path: Path,
    output: Path,
    verbose: bool = True,
    downscale_factor: int = 1,
    run_sparse: bool = True,
    run_dense: bool = True,
    transform_poses: bool = True,
):
    assert path.exists()
    # Also, Ignatius may have corrupted intrinsics file if downloaded from NSVF
    intrinsics = np.loadtxt(path / "intrinsics.txt")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    params = [fx, fy, cx, cy]
    width, height = Image.open(next(iter((path / "rgb").glob("*.png")))).size

    training_images = list((path / "rgb").glob("0_*.png"))
    testing_images = list((path / "rgb").glob("1_*.png"))

    # Process cameras.bin
    output.mkdir(parents=True, exist_ok=True)
    (output / "colmap" / "sparse").mkdir(parents=True, exist_ok=True)
    colmap_utils.write_cameras_binary(
        {1: colmap_utils.Camera(1, "PINHOLE", width, height, params)},
        output / "colmap" / "sparse" / "cameras.bin",
    )

    # Downscale images and copy to output directory
    with status("[bold yellow]Downscaling images...", visible=not verbose):
        assert isinstance(downscale_factor, int)
        downscale_dir = output / f"images_{downscale_factor}"
        downscale_dir.mkdir(parents=True, exist_ok=True)
        for f in itertools.chain(training_images, testing_images):
            out = downscale_dir / f.name
            if not out.exists():
                # Downscale and add white background (only for Ignatius which is corrupted in the NSVF download)
                downscale_cmd = ""
                if downscale_factor > 1:
                    downscale_cmd = f"scale=iw/{downscale_factor}:ih/{downscale_factor}[0];[0]"
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-noautorotate",
                    "-i",
                    str(f),
                    "-q:v",
                    "2",
                    "-filter_complex",
                    f"[v:0]{downscale_cmd}split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto",
                    "-frames:v",
                    "1",
                    str(out),
                ]
                run_command(ffmpeg_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    CONSOLE.log(f"Images saved to {downscale_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        with status(status="[bold yellow]Preparing training images...", visible=not verbose):
            train_images_path = Path(tmpdir) / "training_images"
            train_images_path.mkdir(parents=True, exist_ok=True)
            for image_path in training_images:
                shutil.copy(image_path, train_images_path / image_path.name)

        # Rescale
        old_w, old_h = width, height
        width = width // downscale_factor
        height = height // downscale_factor
        fx = fx * width / old_w
        fy = fy * height / old_h
        cx = cx * width / old_w
        cy = cy * height / old_h
        # Export poses for MinimalDataLoader

        training_images.sort()
        testing_images.sort()

        c2ws = []
        for f in itertools.chain(training_images, testing_images):
            c2w = np.loadtxt(f.parent.parent / "pose" / (f.stem + ".txt"))
            c2w[0:3, 1:3] *= -1
            c2ws.append(c2w)
        c2ws = np.stack(c2ws, 0)

        out = {}
        if transform_poses:
            c2ws, applied_transform, applied_scale = _transform_poses(c2ws)
            out["applied_transform"] = applied_transform
            out["applied_scale"] = applied_scale

        for split in ["train", "val", "test"]:
            file_list = training_images if split == "train" else testing_images
            cameras = c2ws[: len(training_images)] if split == "train" else c2ws[len(training_images) :]

            out_fname = str(output / f"{split}.npz")
            np.savez(
                out_fname,
                image_filenames=[f"images_{downscale_factor}/{x.name}" for x in file_list],
                scene_box=np.loadtxt(path / "bbox.txt")[:6].reshape((2, 3)),
                cameras=np.array(
                    {
                        "fx": np.array(fx, np.float32),
                        "fy": np.array(fy, np.float32),
                        "cx": np.array(cx, np.float32),
                        "cy": np.array(cy, np.float32),
                        "width": np.array(width, np.int64),
                        "height": np.array(height, np.int64),
                        "camera_to_worlds": np.stack(cameras, 0).astype(np.float32),
                        "camera_type": np.array(1, np.int32),  # 1 is perspective camera
                    }
                ),
                **out,
            )
            CONSOLE.log(f"Split {split} saved to {out_fname}")

        if run_sparse:
            with status(status="[bold yellow]Generating sparse model...", visible=not verbose):
                run_command(
                    [
                        "colmap",
                        "database_creator",
                        "--database_path",
                        output / "colmap" / "database.db",
                    ],
                    verbose=verbose,
                )
                run_command(
                    [
                        "colmap",
                        "feature_extractor",
                        "--database_path",
                        output / "colmap" / "database.db",
                        "--image_path",
                        train_images_path,
                        "--ImageReader.single_camera",
                        "1",
                        "--ImageReader.camera_model",
                        "PINHOLE",
                        "--ImageReader.camera_params",
                        ",".join(map(str, params)),
                    ],
                    verbose=verbose,
                )
                run_command(
                    [
                        "colmap",
                        "exhaustive_matcher",
                        "--database_path",
                        output / "colmap" / "database.db",
                    ],
                    verbose=verbose,
                )

                db = colmap_utils.COLMAPDatabase(output / "colmap" / "database.db")
                db_images = db.execute("select * from images").fetchall()
                id_map = {x[1]: x[0] for x in db_images}

                images = {}
                for f in sorted(training_images):
                    c2w = np.loadtxt(f.parent.parent / "pose" / (f.stem + ".txt"))
                    w2c = np.linalg.inv(c2w)
                    qvec = colmap_utils.rotmat2qvec(w2c[:3, :3])
                    tvec = w2c[:3, 3]
                    images[id_map[f.name]] = colmap_utils.Image(id_map[f.name], qvec, tvec, 1, f.name, [], [])
                colmap_utils.write_images_binary(images, output / "colmap" / "sparse" / "images.bin")
                colmap_utils.write_points3D_binary({}, output / "colmap" / "sparse" / "points3D.bin")

                run_command(
                    [
                        "colmap",
                        "point_triangulator",
                        "--clear_points",
                        "1",
                        "--database_path",
                        output / "colmap" / "database.db",
                        "--input_path",
                        output / "colmap" / "sparse",
                        "--output_path",
                        output / "colmap" / "sparse",
                        "--image_path",
                        train_images_path,
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "model_converter",
                        "--input_path",
                        output / "colmap" / "sparse",
                        "--output_path",
                        output / "colmap" / "sparse" / "pointcloud.ply",
                        "--output_type",
                        "PLY",
                    ],
                    verbose=verbose,
                )
                shutil.copy(output / "colmap/sparse/pointcloud.ply", output / "sparse.ply")
                CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{output/'sparse.ply'}")

        if run_dense and not (output / "dense.ply").exists():
            with status(status="[bold yellow]Generating dense model...", visible=not verbose):
                run_command(
                    [
                        "colmap",
                        "image_undistorter",
                        "--input_path",
                        output / "colmap" / "sparse",
                        "--output_path",
                        output / "colmap" / "dense",
                        "--image_path",
                        train_images_path,
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "patch_match_stereo",
                        "--workspace_path",
                        output / "colmap" / "dense",
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "stereo_fusion",
                        "--output_path",
                        output / "colmap" / "dense" / "fused.ply",
                        "--workspace_path",
                        output / "colmap" / "dense",
                    ],
                    verbose=verbose,
                )

                shutil.copy(output / "colmap/dense/fused.ply", output / "dense.ply")
                CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{output/'dense.ply'}")


if __name__ == "__main__":
    tyro.cli(entrypoint)

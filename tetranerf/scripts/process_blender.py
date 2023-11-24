import contextlib
import json
import math
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
    transforms: Path,
    output: Path,
    verbose: bool = True,
    run_sparse: bool = True,
    run_dense: bool = False,
):
    output.mkdir(parents=True, exist_ok=True)
    with transforms.open("r") as f:
        data = json.load(f)

    frames = data.pop("frames")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpoutput = Path(tmpdir)
        # Copy images
        rename_map = {}
        with status(status="[bold yellow]Copying images...", visible=not verbose):
            (tmpoutput / "images").mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                source: Path = transforms.parent / (fr["file_path"] + ".png")
                target = tmpoutput / "images" / f"frame_{i+1:05d}.png"
                rename_map[source.name] = target.name
                if not target.exists():
                    shutil.copy(source, target)
            CONSOLE.print(f"[green bold]Images copied to [yellow]{tmpoutput/'images'}")

        # Process cameras.bin
        (output / "sparse").mkdir(parents=True, exist_ok=True)
        img = Image.open(transforms.parent / (frames[0]["file_path"] + ".png"))
        width, height = img.size
        cx = width / 2
        cy = height / 2
        focal = 0.5 * width / math.tan(0.5 * data["camera_angle_x"])
        params = [focal, cx, cy]
        colmap_utils.write_cameras_binary(
            {1: colmap_utils.Camera(1, "SIMPLE_PINHOLE", width, height, params)},
            output / "sparse" / "cameras.bin",
        )

        if run_sparse:
            with status(status="[bold yellow]Generating sparse model...", visible=not verbose):
                run_command(
                    [
                        "colmap",
                        "database_creator",
                        "--database_path",
                        output / "sparse" / "database.db",
                    ],
                    verbose=verbose,
                )
                run_command(
                    [
                        "colmap",
                        "feature_extractor",
                        "--database_path",
                        output / "sparse" / "database.db",
                        "--image_path",
                        tmpoutput / "images",
                        "--ImageReader.single_camera",
                        "1",
                        "--ImageReader.camera_model",
                        "SIMPLE_PINHOLE",
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
                        output / "sparse" / "database.db",
                    ],
                    verbose=verbose,
                )

                db = colmap_utils.COLMAPDatabase(output / "sparse" / "database.db")
                db_images = db.execute("select * from images").fetchall()
                id_map = {x[1]: x[0] for x in db_images}

                images = {}
                for i, fr in enumerate(frames):
                    c2w = np.array(fr["transform_matrix"])
                    c2w[0:3, 1:3] *= -1
                    w2c = np.linalg.inv(c2w)

                    qvec = colmap_utils.rotmat2qvec(w2c[:3, :3])
                    tvec = w2c[:3, 3]
                    name = Path(fr["file_path"]).name + ".png"
                    name = rename_map.get(name, name)
                    images[id_map[name]] = colmap_utils.Image(id_map[name], qvec, tvec, 1, name, [], [])
                colmap_utils.write_images_binary(images, output / "sparse" / "images.bin")
                colmap_utils.write_points3D_binary({}, output / "sparse" / "points3D.bin")

                run_command(
                    [
                        "colmap",
                        "point_triangulator",
                        "--clear_points",
                        "1",
                        "--database_path",
                        output / "sparse" / "database.db",
                        "--input_path",
                        output / "sparse",
                        "--output_path",
                        output / "sparse",
                        "--image_path",
                        tmpoutput / "images",
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "model_converter",
                        "--input_path",
                        output / "sparse",
                        "--output_path",
                        output / "sparse" / "pointcloud.ply",
                        "--output_type",
                        "PLY",
                    ],
                    verbose=verbose,
                )
                shutil.copy(output / "sparse/pointcloud.ply", output / "sparse.ply")
                CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{output/'sparse.ply'}")

        if run_dense:
            with status(status="[bold yellow]Generating dense model...", visible=not verbose):
                run_command(
                    [
                        "colmap",
                        "image_undistorter",
                        "--input_path",
                        output / "sparse",
                        "--output_path",
                        output / "dense",
                        "--image_path",
                        tmpoutput / "images",
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "patch_match_stereo",
                        "--workspace_path",
                        output / "dense",
                    ],
                    verbose=verbose,
                )

                run_command(
                    [
                        "colmap",
                        "stereo_fusion",
                        "--output_path",
                        output / "dense" / "fused.ply",
                        "--workspace_path",
                        output / "dense",
                    ],
                    verbose=verbose,
                )

                shutil.copy(output / "dense/fused.ply", output / "dense.ply")
                CONSOLE.print(f"[green bold]Sparse model stored in [yellow]{output/'dense.ply'}")


if __name__ == "__main__":
    tyro.cli(entrypoint)

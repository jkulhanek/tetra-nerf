from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nerfstudio.utils
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components import renderers
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from rich.console import Console
from skimage.metrics import structural_similarity
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ..utils.extension import TetrahedraTracer, interpolate_values

CONSOLE = Console(width=120)

try:
    import dm_pix as pix
    import jax

    jax_ssim = jax.jit(pix.ssim)

    def mipnerf_ssim(image, rgb):
        values = [
            float(jax_ssim(gt, img))
            for gt, img in zip(image.cpu().permute(0, 2, 3, 1).numpy(), rgb.cpu().permute(0, 2, 3, 1).numpy())
        ]
        return sum(values) / len(values)

except ImportError:
    CONSOLE.print("[yellow]JAX not installed, skipping Mip-NeRF SSIM[/yellow]")
    mipnerf_ssim = None


def skimage_ssim(image, rgb):
    # Scikit implementation used in PointNeRF
    values = [
        structural_similarity(gt, img, win_size=11, multichannel=True, channel_axis=2, data_range=1.0)
        for gt, img in zip(image.cpu().permute(0, 2, 3, 1).numpy(), rgb.cpu().permute(0, 2, 3, 1).numpy())
    ]
    return sum(values) / len(values)


@dataclass
class TetrahedraNerfConfig(ModelConfig):
    _target: Any = dataclasses.field(default_factory=lambda: TetrahedraNerf)
    tetrahedra_path: Optional[Path] = None
    num_tetrahedra_vertices: Optional[int] = None
    num_tetrahedra_cells: Optional[int] = None

    max_intersected_triangles: int = 512  # TODO: try 1024
    num_samples: int = 256
    num_fine_samples: int = 256
    field_dim: int = 64

    num_color_layers: int = 1
    num_density_layers: int = 3
    hidden_size: int = 128

    input_fourier_frequencies: int = 0

    initialize_colors: bool = True

    def __post_init__(self):
        if self.tetrahedra_path is not None and self.num_tetrahedra_vertices is None:
            if not self.tetrahedra_path.exists():
                raise RuntimeError(f"Tetrahedra path {self.tetrahedra_path} does not exist")
            tetrahedra = torch.load(self.tetrahedra_path)
            self.num_tetrahedra_vertices = len(tetrahedra["vertices"])
            self.num_tetrahedra_cells = len(tetrahedra["cells"])


# pylint: disable=attribute-defined-outside-init
class TetrahedraNerf(Model):
    """Tetrahedra NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: TetrahedraNerfConfig

    def __init__(
        self,
        config: TetrahedraNerfConfig,
        dataparser_transform=None,
        dataparser_scale=None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )

        self.dataparser_transform = dataparser_transform
        self.dataparser_scale = dataparser_scale
        self._tetrahedra_tracer = None
        if self.config.num_tetrahedra_vertices is None:
            raise RuntimeError("The tetrahedra_path must be specified.")
        self.register_buffer(
            "tetrahedra_vertices",
            torch.empty((self.config.num_tetrahedra_vertices, 3), dtype=torch.float32),
        )
        self.register_buffer(
            "tetrahedra_cells",
            torch.empty((self.config.num_tetrahedra_cells, 4), dtype=torch.int32),
        )
        self.register_parameter(
            "tetrahedra_field",
            nn.Parameter(
                torch.empty(
                    (self.config.field_dim, self.config.num_tetrahedra_vertices),
                    dtype=torch.float32,
                )
            ),
        )
        self._tetrahedra_initialized = False

    # Just to allow for size reduction of the checkpoint
    def load_state_dict(self, state_dict, strict: bool = True):
        for k, v in self.lpips.state_dict().items():
            state_dict[f"lpips.{k}"] = v
        if hasattr(self, "lpips_vgg"):
            for k, v in self.lpips_vgg.state_dict().items():
                state_dict[f"lpips_vgg.{k}"] = v
        return super().load_state_dict(state_dict, strict)

    # Just to allow for size reduction of the checkpoint
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        for k in self.lpips.state_dict():
            state_dict.pop(f"lpips.{k}")
        if hasattr(self, "lpips_vgg"):
            for k in self.lpips_vgg.state_dict():
                state_dict.pop(f"lpips_vgg.{k}")
        return state_dict

    @staticmethod
    def _init_tetrahedra_field(tetrahedra_field):
        scale = 1e-4
        tetrahedra_field.uniform_(-scale, scale)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        will_initialize = False
        if (
            f"{prefix}tetrahedra_vertices" in state_dict
            and f"{prefix}tetrahedra_cells" in state_dict
            and f"{prefix}tetrahedra_field" in state_dict
        ):
            will_initialize = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if will_initialize:
            self._tetrahedra_initialized = True

    def _init_tetrahedra(self):
        if self.config.tetrahedra_path is not None:
            if not self.config.tetrahedra_path.exists():
                raise RuntimeError(f"Specified tetrahedra path {self.config.tetrahedra_path} does not exist")
            tetrahedra = torch.load(str(self.config.tetrahedra_path), map_location=torch.device("cpu"))
            tetrahedra_vertices = tetrahedra["vertices"].float()

            # Transform vertices using the dataparser transforms
            if self.dataparser_scale is None:
                raise RuntimeError(
                    "Could not read the dataparser_scale and dataparser_transform parameters."
                    "Make sure you are using the TetrahedraNerfPipeline with the model."
                )

            tetrahedra_vertices = (
                torch.cat(
                    (
                        tetrahedra_vertices,
                        torch.ones_like(tetrahedra_vertices[..., :1]),
                    ),
                    -1,
                )
                @ self.dataparser_transform.T
            )
            tetrahedra_vertices *= self.dataparser_scale

            tetrahedra_cells = tetrahedra["cells"].int()
            num_tetrahedra_vertices = len(tetrahedra_vertices)
            self.tetrahedra_vertices.copy_(tetrahedra_vertices.to(device=self.tetrahedra_vertices.device))
            self.tetrahedra_cells.copy_(tetrahedra_cells.to(device=self.tetrahedra_cells.device))
            self._init_tetrahedra_field(self.tetrahedra_field.data)
            if self.config.initialize_colors:
                assert "colors" in tetrahedra
                assert tetrahedra["colors"].dtype == torch.uint8
                assert tetrahedra["colors"].shape == (num_tetrahedra_vertices, 4)
                colors = tetrahedra["colors"].float().to(self.tetrahedra_field.device) * 2.0 / 255.0 - 1.0
                self.tetrahedra_field.data[1:4, :] = colors[:, :3].T
                self.tetrahedra_field.data[0, :] = colors[:, 3]
            CONSOLE.print(f"Tetrahedra initialized from file {self.config.tetrahedra_path}:")
            CONSOLE.print(f"    Num points: {len(self.tetrahedra_vertices)}")
            CONSOLE.print(f"    Num tetrahedra: {len(self.tetrahedra_cells)}")
            self._tetrahedra_initialized = True
        else:
            raise RuntimeError("The tetrahedra_path must be specified.")

    def get_tetrahedra_tracer(self):
        device = self.tetrahedra_field.device
        if device.type != "cuda":
            raise RuntimeError("Tetrahedra tracer is only supported on a CUDA device")
        if self._tetrahedra_tracer is not None:
            if self._tetrahedra_tracer.device == device:
                return self._tetrahedra_tracer
            del self._tetrahedra_tracer
            self._tetrahedra_tracer = None
        if not self._tetrahedra_initialized:
            self._init_tetrahedra()
        self._tetrahedra_tracer = TetrahedraTracer(device)
        self._tetrahedra_tracer.load_tetrahedra(self.tetrahedra_vertices, self.tetrahedra_cells)
        return self._tetrahedra_tracer

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        mlp_in_dim = self.config.field_dim
        if self.config.input_fourier_frequencies > 0:
            self.position_encoding = NeRFEncoding(
                in_dim=self.config.field_dim,
                num_frequencies=self.config.input_fourier_frequencies,
                min_freq_exp=0.0,
                max_freq_exp=float(self.config.input_fourier_frequencies),
                include_input=True,
            )
            mlp_in_dim += self.position_encoding.get_out_dim()
        else:
            self.position_encoding = lambda x: x
        self.direction_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=4.0,
            include_input=True,
        )
        self.mlp_base = MLP(
            in_dim=mlp_in_dim,
            num_layers=self.config.num_density_layers,
            layer_width=self.config.hidden_size,
            out_activation=nn.ReLU(),
        )
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=self.config.num_color_layers,
            layer_width=self.config.hidden_size,
            out_activation=nn.ReLU(),
        )

        self.field_output_color = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_samples)
        if self.config.num_fine_samples > 0:
            self.sampler_pdf = PDFSampler(num_samples=self.config.num_fine_samples)

        # renderers
        self._background_color = nerfstudio.utils.colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=self._background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.skimage_ssim = skimage_ssim
        self.nerfstudio_ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        # self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.parameters())
        return param_groups

    def get_background_color(self):
        background_color = self._background_color
        if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = renderers.BACKGROUND_COLOR_OVERRIDE
        return background_color

    def get_outputs(self, ray_bundle: RayBundle):
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        tracer = self.get_tetrahedra_tracer()
        tracer_output = tracer.trace_rays(
            ray_bundle.origins.contiguous(),
            ray_bundle.directions.contiguous(),
            self.config.max_intersected_triangles,
        )
        num_visited_cells = tracer_output["num_visited_cells"]
        nears = tracer_output["hit_distances"][:, 0, 0][:, None]
        fars = torch.gather(
            tracer_output["hit_distances"][:, :, 1],
            1,
            (num_visited_cells[:, None].long() - 1).clamp_min_(0),
        )

        # Reduce everything to nonempty rays
        ray_mask = tracer_output["num_visited_cells"] > 0
        nears_r = nears[ray_mask]
        fars_r = fars[ray_mask]
        if nears_r.shape[0] > 0:
            ray_bundle_modified_r = dataclasses.replace(ray_bundle[ray_mask], nears=nears_r, fars=fars_r)
            ray_samples_r: RaySamples = self.sampler_uniform(ray_bundle_modified_r)
            distances_r = (ray_samples_r.frustums.ends + ray_samples_r.frustums.starts) / 2

            # Trace matched cells and interpolate field
            traced_cells = tracer.find_visited_cells(
                tracer_output["num_visited_cells"][ray_mask],
                tracer_output["visited_cells"][ray_mask],
                tracer_output["barycentric_coordinates"][ray_mask],
                tracer_output["hit_distances"][ray_mask],
                distances_r.squeeze(-1),
            )
            barycentric_coords = traced_cells["barycentric_coordinates"]
            field_values = interpolate_values(
                traced_cells["vertex_indices"],
                barycentric_coords,
                self.tetrahedra_field,
            )

            if self.config.num_fine_samples > 0:
                # apply MLP on top
                encoded_abc = self.position_encoding(field_values)
                base_mlp_out = self.mlp_base(encoded_abc)

                # Apply dense, fine sampling
                density_coarse = self.field_output_density(base_mlp_out)
                weights = ray_samples_r.get_weights(density_coarse)
                # pdf sampling
                ray_samples_r = self.sampler_pdf(ray_bundle_modified_r, ray_samples_r, weights)
                distances_r = (ray_samples_r.frustums.ends + ray_samples_r.frustums.starts) / 2

                traced_cells = tracer.find_visited_cells(
                    tracer_output["num_visited_cells"][ray_mask],
                    tracer_output["visited_cells"][ray_mask],
                    tracer_output["barycentric_coordinates"][ray_mask],
                    tracer_output["hit_distances"][ray_mask],
                    distances_r.squeeze(-1),
                )
                barycentric_coords = traced_cells["barycentric_coordinates"]
                field_values = interpolate_values(
                    traced_cells["vertex_indices"],
                    barycentric_coords,
                    self.tetrahedra_field,
                )

            encoded_abc = self.position_encoding(field_values)
            base_mlp_out = self.mlp_base(encoded_abc)

            field_outputs = {}
            field_outputs[self.field_output_density.field_head_name] = self.field_output_density(base_mlp_out)
            encoded_dir = self.direction_encoding(ray_samples_r.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, base_mlp_out], dim=-1))  # type: ignore
            field_outputs[self.field_output_color.field_head_name] = self.field_output_color(mlp_out)

            weights = ray_samples_r.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_r = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.RGB],
                weights=weights,
            )
            accumulation_r = self.renderer_accumulation(weights)
            depth_r = self.renderer_depth(weights, ray_samples_r)

        # Expand rendered values back to the original shape
        device = ray_mask.device
        rgb = (
            self.get_background_color()
            .to(device=device, dtype=torch.float32)
            .view(1, 3)
            .repeat_interleave(ray_mask.shape[0], 0)
        )
        # rgb = torch.zeros((ray_mask.shape[0], 3), dtype=torch.float32, device=device)
        accumulation = torch.zeros((ray_mask.shape[0], 1), dtype=torch.float32, device=device)
        depth = torch.full(
            (ray_mask.shape[0], 1),
            self.collider.far_plane,
            dtype=torch.float32,
            device=device,
        )
        if nears_r.shape[0] > 0:
            rgb[ray_mask] = rgb_r
            accumulation[ray_mask] = accumulation_r
            depth[ray_mask] = depth_r

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "ray_mask": ray_mask,
        }
        return outputs

    # pylint: disable=unused-argument
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "lpips": float(lpips),
            # "lpips_vgg": float(self.lpips_vgg(image, rgb)),
            "nerfstudio_ssim": float(self.nerfstudio_ssim(image, rgb)),
            "skimage_ssim": float(self.skimage_ssim(image, rgb)),
            "lpips": float(lpips),
        }
        if mipnerf_ssim is not None:
            metrics_dict["mipnerf_ssim"] = float(mipnerf_ssim(image, rgb))

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        return metrics_dict, images_dict

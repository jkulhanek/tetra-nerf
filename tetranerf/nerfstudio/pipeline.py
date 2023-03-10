import typing

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from typing_extensions import Literal


class TetrahedraNerfPipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        Pipeline.__init__(self)
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # Loaded pointcloud must be transformed using the transform from the Dataparser
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            dataparser_transform=self.datamanager.train_dataparser_outputs.dataparser_transform,
            dataparser_scale=self.datamanager.train_dataparser_outputs.dataparser_scale,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

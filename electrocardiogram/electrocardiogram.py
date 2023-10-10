import collections
from typing import Mapping
from tensorflow_datasets.core import dataset_collection_builder
from tensorflow_datasets.core import naming


class Electrocardiogram(dataset_collection_builder.DatasetCollection):
    """Dataset collection builder my_dataset_collection."""

    @property
    def info(self) -> dataset_collection_builder.DatasetCollectionInfo:
        return dataset_collection_builder.DatasetCollectionInfo.from_cls(
            dataset_collection_class=self.__class__,
            description="The collection of electrocardiogram datasets.",
            release_notes={
                "1.0.0": "Initial release",
            },
        )

    @property
    def datasets(
            self,
    ) -> Mapping[str, Mapping[str, naming.DatasetReference]]:
        return collections.OrderedDict({
            "1.0.0":
                naming.references_for({
                    "zheng": "zheng:1.0.2",
                    "ptb": "ptb:1.0.1",
                    "icentia": "icentia11k:1.0.2",
                    "synth": "synth:1.0.0",
                    "medalcare": "medalcare:1.0.0",
                })
        })

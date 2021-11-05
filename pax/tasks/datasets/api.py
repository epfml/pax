from typing import Iterable


class Batch:
    progress: float

    def __len__(self) -> int:
        return len(self.x)


class Dataset:
    def iterator(
        self,
        batch_size: int,
        shuffle: bool,
        repeat: bool = False,
        drop_last: bool = True,
        num_workers: int = 1,
    ) -> Iterable[Batch]:
        pass

    def __len__(self) -> int:
        pass

    @property
    def num_classes(self) -> int:
        pass

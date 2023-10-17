from typing import TypeVar

from .data_types import skip
from .reusable import reusable
from .stream import Stream
from ..video import Video


T = TypeVar("T")


@reusable
class PruneFrames(Stream[T]):
    def __init__(self, prunner: Stream[bool], stream: Stream[T]):
        self.prunner = prunner
        self.stream = stream

    def stream(self, video: Video):
        for prune, frame in zip(self.prunner.stream(video), self.stream.stream(video)):
            if prune is True and frame != skip:
                yield frame
            else:
                yield skip

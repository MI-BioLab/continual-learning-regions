from typing import Sequence

def tile_elements(elements: Sequence,
                  times: int):
    return [elem for elem in elements for _ in range(times)]

__all__ = ["tile_elements"]
"""Bounded host-side frame cache using the official SAM3 pixel preprocessing."""
from collections import OrderedDict


class BoundedFrameLoader:
    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        if not offload_video_to_cpu:
            raise ValueError('BoundedFrameLoader requires CPU frame offloading')
        self.img_paths = img_paths
        self.image_size = image_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.cache = OrderedDict()
        self.video_height = self.video_width = None
        self[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        import torch
        from sam3.model.io_utils import _load_img_as_tensor
        index = int(index)
        if index not in self.cache:
            image, height, width = _load_img_as_tensor(self.img_paths[index], self.image_size)
            self.video_height, self.video_width = height, width
            self.cache[index] = (image.to(torch.float16) - self.img_mean) / self.img_std
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def install_bounded_loader():
    # Scoped to the isolated worker. Model, tracker and checkpoint code stay untouched.
    # The official async loader eagerly fills all host RAM; we implement the same
    # indexing interface with eight frames, retaining the entire temporal model state.
    from sam3.model import io_utils
    io_utils.AsyncImageFrameLoader = BoundedFrameLoader

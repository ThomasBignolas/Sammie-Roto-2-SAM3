"""Small CPU cache retaining Sammie's exact SAM 2 preprocessing and float32 dtype."""
from collections import OrderedDict


class BoundedSam2Frames:
    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std, compute_device):
        from sammie.performance import memory_budget
        self.paths, self.size = img_paths, image_size
        self.mean, self.std = img_mean.cpu(), img_std.cpu()
        self.offload, self.device = offload_video_to_cpu, compute_device
        self.limit = min(4, memory_budget(48) // (3 * image_size**2 * 4))
        self.cache = OrderedDict()
        self[0]

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        from sam2.utils.misc import _load_img_as_tensor
        index = int(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            image = self.cache[index]
        else:
            image, self.video_height, self.video_width = _load_img_as_tensor(self.paths[index], self.size)
            image -= self.mean
            image /= self.std
            if self.limit:
                self.cache[index] = image
                while len(self.cache) > self.limit: self.cache.popitem(last=False)
        return image if self.offload else image.to(self.device, non_blocking=True)


def install():
    from sam2.utils import misc
    misc.AsyncVideoFrameLoader = BoundedSam2Frames

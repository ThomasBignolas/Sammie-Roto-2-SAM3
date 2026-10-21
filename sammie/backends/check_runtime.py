"""Checkpoint-free validation of CUDA, official imports and required Triton kernels."""
import json


def main():
    from .worker import configure_runtime
    configure_runtime()
    import numpy
    import torch
    import torchvision
    from sam3.model_builder import build_sam3_video_predictor, build_sam3_multiplex_video_predictor
    from sam3.model.edt import edt_triton
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the isolated SAM3 environment')
    tensor = torch.zeros((1, 8, 8), dtype=torch.bool, device='cuda')
    edt_triton(tensor)
    torch.cuda.synchronize()
    print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'numpy': numpy.__version__, 'gpu': torch.cuda.get_device_name(),
                      'bf16': torch.cuda.is_bf16_supported(), 'triton_edt': 'passed'}))


if __name__ == '__main__':
    main()

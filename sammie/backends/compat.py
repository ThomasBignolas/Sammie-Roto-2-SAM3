"""Narrow compatibility fixes for the pinned official SAM3 source on Windows."""


def configure_attention():
    import torch
    if torch.backends.cuda.is_flash_attention_available():
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from sam3.model import decoder

    def available_sdpa(backends, **kwargs):
        # decoder.functional_attention hardcodes FLASH_ATTENTION even with use_fa3=False.
        # Use the same PyTorch attention operation, permitting native Windows kernels.
        # No attention approximation, quantization or change to temporal memory.
        return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.MATH], **kwargs)

    decoder.sdpa_kernel = available_sdpa

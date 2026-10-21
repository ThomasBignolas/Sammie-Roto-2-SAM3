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


def configure_multiplex_output(model):
    """Preserve tracker results on frames not previously visited by the detector.

    Pinned upstream _build_sam2_output returns {} before merging refinements when
    the frame is absent from cached_frame_outputs. A selected-object-only pass
    legitimately reaches uncached frames. Seed an empty cache entry so the
    official merge and subsequent suppression logic both execute normally.
    """
    original = model._build_sam2_output

    def merge(state, frame, refined_obj_id_to_mask=None):
        if refined_obj_id_to_mask is not None:
            state['cached_frame_outputs'].setdefault(frame, {})
        return original(state, frame, refined_obj_id_to_mask)

    model._build_sam2_output = merge

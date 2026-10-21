# Source and model licenses

This integration retains the original repository's [GPL-3.0 license](../LICENSE) and attribution. It does not relicense third-party components. No downloaded SAM/MatAnyone checkpoints, account tokens, or official SAM 3 source files are added to Git. The small mask-smoothing weight already bundled by upstream remains unchanged.

| Component | Relevant terms / access |
| --- | --- |
| Sammie-Roto 2 | [Upstream GPL-3.0](https://github.com/Zarxrax/Sammie-Roto-2/blob/e759816843a741d25411db7f360b0ce286259d76/LICENSE). Existing notices remain. No CONTRIBUTING file was present at the inspected upstream revision. Work stays on the development branch/repository; upstream is not modified. |
| Bundled SAM 2 | [Apache-2.0](../sam2/LICENSE), plus retained third-party notices. Original download mechanism is preserved. |
| SAM 3 / SAM 3.1 source and checkpoints | [Meta SAM License](https://github.com/facebookresearch/sam3/blob/7f473cfd983d5eb43040ff608f66bff0d7bdcb53/LICENSE), last updated November 19, 2025. The source's MIT package classifier does not replace this license. The agreement covers model code and weights, has use restrictions, and requires a copy of the agreement when redistributing covered materials. Users obtain the source/checkpoints directly from Meta and accept the gated model conditions through their own Hugging Face account. |
| MatAnyone | [S-Lab License 1.0](../matanyone/LICENSE). The included license permits non-commercial use and asks users to contact its contributors for commercial permission. This matters for paid VFX work; the integration does not grant such permission. |
| ViTMatte | Original [HUST Vision Lab source is MIT](https://github.com/hustvl/ViTMatte/blob/8cd7ef068380977c3962c4cb733cb1fe7f2241a5/LICENSE). The official organization's [Small Composition-1k converted checkpoint card](https://huggingface.co/hustvl/vitmatte-small-composition-1k/blob/6a58ad7646403c1df626fbd746900aec7361ea1d/README.md) declares Apache-2.0 (card written by the Hugging Face team). The Transformers implementation is Apache-2.0. This integration downloads public weights locally; no checkpoint is redistributed. A custom checkpoint's own terms remain applicable. |
| Other original models | Their original model-specific notices and access terms remain applicable. This upgrade does not expand those permissions. |

Official model pages: [SAM 3](https://huggingface.co/facebook/sam3), [SAM 3.1](https://huggingface.co/facebook/sam3.1). Approval on one does not substitute for access to the other. Redistribution of an environment containing Meta's package or checkpoints requires following Meta's terms; the installer downloading them locally is separate from distributing this application source.

Changes to Meta's repository, if proposed upstream in the future, should follow its [contribution requirements](https://github.com/facebookresearch/sam3/blob/7f473cfd983d5eb43040ff608f66bff0d7bdcb53/CONTRIBUTING.md). This development repository does not submit or publish an upstream contribution automatically.

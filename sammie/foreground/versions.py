"""Public CK1.0 only. Sources and weights are installed locally, never vendored."""
from pathlib import Path
from sammie.aov.cache import fingerprint

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ('nikopueringer/CorridorKey', '97e55a453060745bead1befd293f6e523c4b845c')
SOURCE_PATH = ROOT/'.runtime'/'corridorkey-src'
WEIGHTS = {
    'blue': ('nikopueringer/CorridorKeyBlue_1.0', '51e6ccaa4b703f54be20a72ac2c37784fb9ba1cd',
             'CorridorKeyBlue_1.0.safetensors', '43bc5f6a08a9e5effe5d633d0d84bb0aff91037b35ab85d16cd812b38c5cac23'),
    'green': ('nikopueringer/CorridorKey_v1.0', 'f6386ddf042d8e92aeb5fd16cb9b101cff508195',
              'CorridorKey_v1.0.safetensors', '74d614f7d92fc559a118c30a7deadedc3cacd8ef83dcb85a030d0bed7af8b20b'),
}


def ensure_weights(screen, progress=lambda text: None):
    repo, revision, filename, sha = WEIGHTS[screen]
    folder = ROOT/'.runtime'/'corridorkey-models'/screen/revision
    path = folder/filename
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        progress(f'Downloading CorridorKey {screen} model (~300 MB)...')
        hf_hub_download(repo, filename, revision=revision, local_dir=str(folder))
    progress(f'Verifying CorridorKey {screen} checkpoint...')
    if fingerprint(path)['sha256'] != sha:
        raise ValueError(f'CorridorKey checksum mismatch: {path}. Remove this file and run setup again.')
    return path

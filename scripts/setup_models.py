"""Download official gated checkpoints; credentials never enter project files."""
import argparse
from getpass import getpass
import json
from pathlib import Path


def main():
    from huggingface_hub import hf_hub_download, login, get_token
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sam3', 'sam3.1', 'both'], default='both')
    parser.add_argument('--no-login', action='store_true', help='Use an existing HF login or HF_TOKEN')
    parser.add_argument('--login-if-needed', action='store_true', help='Reuse a saved login or already downloaded checkpoints')
    args = parser.parse_args()
    print('Request access and accept Meta terms in your browser:')
    print('https://huggingface.co/facebook/sam3')
    print('https://huggingface.co/facebook/sam3.1')
    print('The login prompt below hides your token. Do not put tokens in source or project settings.')
    target = Path(__file__).resolve().parents[1] / 'checkpoints'
    versions = json.loads((target.parent / 'model_versions.json').read_text(encoding='utf-8'))
    selected = ('sam3', 'sam3.1') if args.model == 'both' else (args.model,)
    def present(model):
        path = target / versions[model]['filename']
        return path.is_file() and path.stat().st_size == versions[model]['bytes']
    need_login = not args.login_if_needed or (not get_token() and not all(present(m) for m in selected))
    if not args.no_login and need_login:
        # Passing token explicitly prevents huggingface_hub's interactive helper
        # from asking again whether it should add a Git credential.
        login(token=getpass('Hugging Face read token (hidden): ').strip(), add_to_git_credential=False)
    for model in selected:
        spec = versions[model]
        name = spec['filename']
        if args.login_if_needed and present(model):
            print(f'Ready (local file size checked): {target / name}')
            continue
        try:
            path = hf_hub_download(spec['repo'], name, revision=spec['revision'], local_dir=target)
            if Path(path).stat().st_size != spec['bytes']:
                raise ValueError('Checkpoint size does not match the pinned official release')
        except Exception as error:
            raise SystemExit(f'{model} download failed ({type(error).__name__}). '
                             'Check approval on the model page and your token read permissions.') from None
        print(f'Ready: {path}')


if __name__ == '__main__':
    main()

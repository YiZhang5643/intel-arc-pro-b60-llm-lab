"""Shared metadata and input provenance for portable benchmarks."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--base-url', default='http://127.0.0.1:5001')
    p.add_argument('--preset', type=Path, help='Declared preset, not proof of running settings')
    return p


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.label):
        raise ValueError('Use a short alphanumeric label')
    if not args.base_url.startswith(('http://', 'https://')):
        raise ValueError('Expected HTTP(S) base URL')
    args.base_url = args.base_url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=False)
    declared = json.loads(args.preset.read_text(encoding='utf-8')) if args.preset else None
    with urllib.request.urlopen(args.base_url + '/props', timeout=10) as f:
        props = json.load(f)
    defaults = props.get('default_generation_settings', {})
    # Preserve only an explicit non-personal settings allowlist; never dump full props.
    observed = {k: defaults[k] for k in ('n_ctx', 'n_predict', 'temperature', 'top_k', 'top_p', 'min_p', 'seed') if k in defaults}
    meta = {'label': args.label, 'declared_preset': declared,
            'preset_verified_against_running_process': False,
            'observed_generation_defaults': observed,
            'observed_slots': props.get('total_slots'),
            'note': 'Keep server launch/version/hardware evidence separately; declaration alone is not verification.'}
    (args.output / 'run-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return args.output

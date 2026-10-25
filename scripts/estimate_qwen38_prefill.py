"""Optimistic compute-only prefill ceiling, NOT a measured performance forecast.

Counts main-model linear projections plus ideal causal QK/AV. Omits delta-net
state/chunk math, MTP, dequantization, memory traffic and scheduling costs.
FMA = 2 FLOPs. Assumes all counted work reaches the FP16 XMX peak.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(metadata, tokens, peak_tflops):
    m = metadata['gguf_inspection']['metadata']
    if m['general.architecture'] != 'qwen35':
        raise ValueError('This estimator only describes the inspected qwen35 graph')
    layers = m['qwen35.block_count'] - m['qwen35.nextn_predict_layers']
    interval = m['qwen35.full_attention_interval']
    if layers % interval:
        raise ValueError('Expected complete full-attention intervals')
    full = layers // interval
    recurrent = layers - full
    d = m['qwen35.embedding_length']
    ffn = m['qwen35.feed_forward_length']
    hq = m['qwen35.attention.head_count']
    hkv = m['qwen35.attention.head_count_kv']
    dk = m['qwen35.attention.key_length']
    dv = m['qwen35.attention.value_length']
    state = m['qwen35.ssm.state_size']
    keys = state * m['qwen35.ssm.group_count']
    values = state * m['qwen35.ssm.time_step_rank']
    vh = m['qwen35.ssm.time_step_rank']
    # Q includes a second, equal-sized gating projection.
    components = {
        'ffn': 2 * layers * 3 * d * ffn,
        'full_attention_projections': 2 * full * d * (2*hq*dk + hkv*dk + hkv*dv + hq*dv),
        'recurrent_projections': 2 * recurrent * d * (2*keys + 3*values + 2*vh),
    }
    linear = sum(components.values())
    rows = []
    for n in tokens:
        # n(n+1)/2 causal pairs, hq heads: QK and AV, each FMA = 2.
        attention = full * hq * (dk + dv) * (n + 1)
        cost = linear + attention
        ceiling = peak_tflops * 1e12 / cost
        rows.append({'tokens': n, 'counted_gflop_per_token': cost/1e9,
                     'ideal_attention_gflop_per_token': attention/1e9,
                     'optimistic_ceiling_tok_s': ceiling,
                     'optimistic_min_prefill_seconds': n/ceiling})
    return {'kind': 'analytical_partial_work_compute_ceiling_not_forecast',
            'model_sha256': metadata['model']['sha256'],
            'assumed_fp16_xmx_tflops': peak_tflops,
            'main_layers': layers, 'full_attention_layers': full,
            'recurrent_layers': recurrent,
            'linear_gflop_per_token': {k: v/1e9 for k, v in components.items()},
            'rows': rows,
            'omitted': ['delta-net state/chunk operations', 'MTP', 'normalization and activations',
                        'dequantization', 'memory traffic', 'padding and masked extra work',
                        'LM head on selected outputs', 'launch/synchronization and host work'],
            'assumptions': ['cold prompt without prefix reuse', 'ideal causal attention skips future pairs',
                            'all counted work runs at FP16 XMX peak; actual kernels may use other paths',
                            'all 64 FFNs counted per token; final-layer output cropping ignored']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, default=ROOT/'results/2026-10-25-qwen38/metadata.json')
    p.add_argument('--tokens', type=int, nargs='+', default=[12288, 32768, 98304])
    p.add_argument('--peak-tflops', type=float, default=98.304)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.peak_tflops <= 0 or any(n <= 0 for n in a.tokens):
        p.error('tokens and peak-tflops must be positive')
    result = estimate(json.loads(a.metadata.read_text(encoding='utf-8')), a.tokens, a.peak_tflops)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if a.output:
        a.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')

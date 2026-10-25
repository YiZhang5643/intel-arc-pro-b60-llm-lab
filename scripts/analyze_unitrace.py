"""Summarize unitrace GPU kernels without double-counting blocking copy spans."""
import argparse
import collections
import csv
import json
from pathlib import Path


def union_us(spans):
    end = None
    total = 0.0
    for a, b in sorted(spans):
        if end is None or a > end:
            total += b-a
        elif b > end:
            total += b-end
        end = b if end is None else max(end, b)
    return total


def category(name):
    if 'MemoryCopy' in name or 'MemoryFill' in name:
        return 'memory_command'
    if 'gemm_kernel' in name:
        return 'gemm'
    if name.startswith('gemm_f16f16f32_nocopy'):
        return 'mkl_gemm'
    if 'micro_sdpa' in name or 'flash_attn' in name or 'mkl_fa_' in name:
        return 'attention'
    if 'gated_delta_net' in name or 'gdn_blocked_sycl' in name:
        return 'gated_delta_net'
    if 'dequantize' in name:
        return 'weight_dequantization'
    if 'convert_unary' in name:
        return 'type_conversion'
    return 'other_kernels'


def analyze(trace, request):
    start = request['client_start_unix_ns']/1000
    end = request['client_end_unix_ns']/1000
    wall_us = end-start
    rows = []
    totals = collections.defaultdict(lambda: {'calls': 0, 'summed_us': 0.0})
    kernels = []
    copies = []
    host = collections.defaultdict(lambda: {'calls': 0, 'summed_us': 0.0})
    for e in trace['traceEvents']:
        if e.get('ph') != 'X':
            continue
        a, b = max(start, e['ts']), min(end, e['ts']+e['dur'])
        if b <= a:
            continue
        if e.get('cat') != 'gpu_op':
            if e['name'].startswith('ze'):
                host[e['name']]['calls'] += 1
                host[e['name']]['summed_us'] += b-a
            continue
        c = category(e['name'])
        totals[c]['calls'] += 1
        totals[c]['summed_us'] += b-a
        rows.append({'category': c, 'start_ms': (a-start)/1000,
                     'duration_ms': (b-a)/1000, 'name': e['name'],
                     'pid': e['pid'], 'tid': e['tid']})
        (copies if c == 'memory_command' else kernels).append((a, b))
    for v in totals.values():
        v['seconds'] = v.pop('summed_us')/1e6
        v['percent_of_client_wall'] = 100*v['seconds']/(wall_us/1e6)
    host_rows = sorted([{'name': k, 'calls': v['calls'], 'seconds': v['summed_us']/1e6}
                        for k, v in host.items()], key=lambda x: -x['seconds'])
    kernel_union = union_us(kernels)
    copy_union = union_us(copies)
    all_union = union_us(kernels+copies)
    return {'client_wall_seconds': wall_us/1e6,
            'kernel_union_seconds': kernel_union/1e6,
            'kernel_union_percent': 100*kernel_union/wall_us,
            'memory_command_union_seconds': copy_union/1e6,
            'memory_commands_outside_kernel_union_seconds': (all_union-kernel_union)/1e6,
            'outside_kernel_union_seconds': (wall_us-kernel_union)/1e6,
            'categories': dict(totals), 'host_api': host_rows,
            'note': 'Memory-command timestamps can include waits and overlap kernels. Do not add their duration to kernel time or infer DMA bandwidth from it. Category times are summed kernel durations, not additive critical-path attribution if kernels overlap.'}, rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    traces = list(a.directory.glob('llama-server.*.json'))
    if len(traces) != 1:
        p.error('Expected exactly one server trace')
    result, rows = analyze(json.loads(traces[0].read_text()),
                          json.loads((a.directory/'profile.json').read_text()))
    (a.directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (a.directory/'kernels.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(result, indent=2))

"""Plot binned kernel activity from analyze_unitrace.py CSV output (requires matplotlib)."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'gemm':'#2878b5', 'attention':'#8958b3', 'gated_delta_net':'#dd8841',
          'weight_dequantization':'#3b9b80', 'type_conversion':'#dfbd46',
          'other_kernels':'#708597', 'mkl_gemm':'#ba679a', 'outside_kernels':'#e0e3e7'}
LABELS = {'gemm':'GEMM', 'attention':'Attention', 'gated_delta_net':'DeltaNet',
          'weight_dequantization':'Weight dequant.', 'type_conversion':'Type conversion',
          'other_kernels':'Other kernels', 'mkl_gemm':'MKL GEMM', 'outside_kernels':'Outside kernels'}


def plot(directories, output):
    categories = ['gemm','attention','gated_delta_net','weight_dequantization',
                  'type_conversion','other_kernels','mkl_gemm','outside_kernels']
    fig, axes = plt.subplots(len(directories), 1, figsize=(12, 2.7*len(directories)), squeeze=False)
    for ax, directory in zip(axes[:,0], directories):
        info = json.loads((directory/'analysis.json').read_text())
        wall = info['client_wall_seconds']
        edges = np.append(np.arange(0, wall, 1.0), wall)
        values = {c:np.zeros(len(edges)-1) for c in categories}
        csv_path = directory/'kernels.csv'
        stream = csv_path.open(newline='', encoding='utf-8') if csv_path.exists() else gzip.open(directory/'kernels.csv.gz', 'rt', newline='', encoding='utf-8')
        with stream as f:
            for r in csv.DictReader(f):
                if r['category']=='memory_command':
                    continue
                a = float(r['start_ms'])/1000
                b = a+float(r['duration_ms'])/1000
                first = max(0, int(a))
                last = min(len(edges)-2, int(b))
                for i in range(first,last+1):
                    values[r['category']][i] += max(0, min(b,edges[i+1])-max(a,edges[i]))
        widths = np.diff(edges)
        active = sum(values[c] for c in categories if c!='outside_kernels')
        if np.any(active-widths > 0.002):
            raise ValueError('Concurrent kernels exceed bin width; use a per-engine chart')
        values['outside_kernels'] = np.maximum(0, widths-active)
        bottom = np.zeros_like(widths)
        for c in categories:
            y = values[c]/widths
            if not np.any(y):
                continue
            ax.bar(edges[:-1], y, width=widths, bottom=bottom, align='edge',
                   color=COLORS[c], linewidth=0, label=LABELS[c])
            bottom += y
        title = '32K default' if '32k-default' in str(directory) else '96K default'
        ax.set_title(f'{title}: {wall:.2f} s; kernel coverage {info["kernel_union_percent"]:.1f}%', loc='left', fontsize=11, weight='bold')
        ax.set_xlim(0,wall)
        ax.set_ylim(0,1)
        ax.set_ylabel('Share of each 1 s bin')
        ax.set_xlabel('Seconds after client request')
        ax.set_yticks([0,0.5,1],['0%','50%','100%'])
        ax.spines[['top','right']].set_visible(False)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Qwen3.8-27B / Arc Pro B60: prefill kernel timeline', y=0.995, fontsize=14, weight='bold')
    fig.text(0.01,0.003,'One profiled cold request per length; MTP3, Q8 KV, ubatch 8192. Blocking copy spans excluded to avoid double counting.', fontsize=8, color='#46505a')
    fig.tight_layout(rect=[0,0.035,1,0.84])
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg']:
        fig.savefig(output.with_suffix(suffix), dpi=160, bbox_inches='tight')
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines()).rstrip()+'\n', encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plot(a.directories,a.output)

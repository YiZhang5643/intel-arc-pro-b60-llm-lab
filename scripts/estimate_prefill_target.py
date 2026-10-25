"""Conditional Amdahl scenarios for the saved Qwen measurements; not speed forecasts."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def estimate(target_tps):
    if target_tps<=0:raise ValueError('target_tps must be positive')
    measured=json.loads((ROOT/'results/2026-10-25-qwen38-kernels/comparison.json').read_text())
    rows=[]
    for group in measured:
        n=group['input_tokens']
        current=next(x for x in group['arms'] if x['blocked']==1)
        trace_path=ROOT/('results/2026-10-25-qwen38-kernels/gdn-profile-32k/analysis.json' if n==32768 else 'results/2026-10-25-qwen38-profile/96k-default/analysis.json')
        categories=json.loads(trace_path.read_text())['categories']
        gemm=categories['gemm']['seconds']
        dequant=categories['weight_dequantization']['seconds']
        now=current['mean_prefill_seconds']
        scenarios={
            'dequantization_eliminated':now-dequant,
            'gemm_twice_as_fast_and_dequantization_eliminated':now-gemm/2-dequant,
            'gemm_and_dequantization_zero_cost':now-gemm-dequant,
        }
        rows.append({'tokens':n,'measured_current_seconds':now,'measured_current_tps':current['aggregate_tokens_per_second'],'target_tps':target_tps,'target_seconds':n/target_tps,'required_throughput_ratio':target_tps/current['aggregate_tokens_per_second'],'gemm_seconds_used':gemm,'dequantization_seconds_used':dequant,'categories_source':str(trace_path.relative_to(ROOT)).replace('\\','/'),'categories_match_new_runtime':n==32768,'conditional_scenarios':{k:{'seconds':v,'tps':n/v} for k,v in scenarios.items()}})
    return {'kind':'conditional_component_time_substitution_not_measured_model_performance','assumptions':['All unmodified component durations remain fixed','32K uses the new blocked-GDN trace; 96K reuses the prior trace GEMM/dequant categories, not a new 96K trace','Category durations are summed kernels and may overlap; estimates are not strict hardware bounds','Quantization, memory pressure, numerical differences and changed graph scheduling are not predicted'],'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-tps',type=float,default=1200)
    p.add_argument('--output',type=Path)
    a=p.parse_args();text=json.dumps(estimate(a.target_tps),indent=2)+'\n'
    if a.output:a.output.write_text(text,encoding='utf-8')
    else:print(text)

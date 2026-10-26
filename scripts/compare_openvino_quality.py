"""Compare scored cold cases only when their full retokenized input hashes match."""
import argparse
import json
from pathlib import Path
import re


def load_cases(directory):
    inputs={}
    for path in directory.glob('input-c*.json'):
        index=re.search(r'input-c(\d+)-',path.name).group(1)
        inputs[index]=json.loads(path.read_text())
    rows={}
    for row in json.loads((directory/'summary.json').read_text()):
        if row.get('generated_tokens',0)<=1:continue
        match=re.search(r'-c(\d+)-r(\d+)$',row['label'])
        if not match:raise ValueError('Expected indexed cold case label')
        provenance=inputs[match.group(1)]
        identity=(provenance['retokenized_token_sha256'],match.group(2))
        if identity in rows:raise ValueError('Duplicate input hash and repeat')
        rows[identity]=(row,provenance)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--fixture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    baseline,candidate=load_cases(a.baseline),load_cases(a.candidate)
    fixtures={}
    for path in a.fixture:
        obj=json.loads(path.read_text());fixtures[obj['retokenized_token_sha256']]=obj
    rows=[]
    for identity,(base,provenance) in baseline.items():
        if identity not in candidate:raise ValueError('Candidate missing a baseline case')
        if identity[0] not in fixtures:raise ValueError('Missing fixture metadata')
        cand,cand_input=candidate[identity];fixture=fixtures[identity[0]]
        if provenance['text_sha256']!=cand_input['text_sha256']:raise ValueError('Input text hashes differ')
        bq,cq=base['quality'],cand['quality']
        if bq['total']!=len(fixture['expected']) or cq['total']!=bq['total']:raise ValueError('Scoring coverage differs')
        bset,cset=set(bq['matched_keys']),set(cq['matched_keys'])
        breakdown={}
        for key,meta in fixture['tasks'].items():
            counts=breakdown.setdefault(meta['type'],{'total':0,'baseline_correct':0,'candidate_correct':0})
            counts['total']+=1;counts['baseline_correct']+=key in bset;counts['candidate_correct']+=key in cset
        rows.append({'seed':fixture['seed'],'input_sha256':identity[0],'text_sha256':provenance['text_sha256'],
            'input_tokens':base['input_tokens'],'repeat':int(identity[1]),'total':bq['total'],
            'baseline_correct':bq['matched'],'candidate_correct':cq['matched'],
            'regressed_keys':sorted(bset-cset),'improved_keys':sorted(cset-bset),'by_task':breakdown,
            'baseline_format_exact_json':bq['format_exact_json'],'candidate_format_exact_json':cq['format_exact_json'],
            'baseline_ttft_seconds':base['callback_first_text_seconds'],
            'candidate_ttft_seconds':cand['callback_first_text_seconds'],
            'baseline_tps':base['effective_tps_to_first_text'],'candidate_tps':cand['effective_tps_to_first_text']})
    obj={'scope':'Two synthetic long-document seeds; not a broad model-quality certification.',
         'cases':rows,'total':sum(r['total'] for r in rows),
         'baseline_correct':sum(r['baseline_correct'] for r in rows),'candidate_correct':sum(r['candidate_correct'] for r in rows),
         'regression_count':sum(len(r['regressed_keys']) for r in rows),
         'improvement_count':sum(len(r['improved_keys']) for r in rows)}
    with a.output.open('x',encoding='utf-8') as file:json.dump(obj,file,indent=2);file.write('\n')
    print(json.dumps(obj,indent=2))


if __name__=='__main__':main()

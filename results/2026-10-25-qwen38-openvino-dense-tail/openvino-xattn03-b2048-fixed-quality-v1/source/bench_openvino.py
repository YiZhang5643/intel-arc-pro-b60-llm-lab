"""Cold long-text OpenVINO IR comparison; run through run_gpu_probe.py for service restoration.

A decomposed VLM uses text input, so the canonical GGUF IDs are decoded and retokenized.
The script records both token hashes and counts, and rejects any changed decoded text.
This compares complete backends/checkpoints, not a controlled quantization-only experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

_overlay_parser=argparse.ArgumentParser(add_help=False)
_overlay_parser.add_argument("--genai-overlay",type=Path)
_overlay_args,_=_overlay_parser.parse_known_args()
if _overlay_args.genai_overlay:sys.path.insert(0,str(_overlay_args.genai_overlay.resolve()))

import openvino as ov
import openvino_genai as genai
from score_long_answer import score_answer


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(',', ':')).encode()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--genai-overlay',type=Path,help='Isolated package with a separately built GenAI DLL')
    p.add_argument('--last-dense-tokens',type=int,help='Dense attention tail during sparse prefill')
    p.add_argument('--prompts',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='GPU.1')
    p.add_argument('--mtp',choices=['on','off'],default='on')
    p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--cache-gb',type=int,default=5)
    p.add_argument('--draft-kv-blocks',type=int,help='Explicit draft KV block capacity')
    p.add_argument('--kv-blocks',type=int)
    p.add_argument('--dq-group',type=int,default=32)
    p.add_argument('--xattention-threshold',type=float,help='Enable GPU XAttention; 1.0 retains all blocks')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--quality-tokens',type=int,default=256)
    p.add_argument('--skip-quality',action='store_true')
    p.add_argument('--quality-in-last-timing',action='store_true',help='Use the last cold timing for retrieval validation, avoiding a duplicate long prefill')
    p.add_argument('--quality-in-all-timings',action='store_true',help='Generate scored answers in every cold timing')
    a=p.parse_args()
    if a.kv_blocks is not None and a.kv_blocks<=0:p.error('kv-blocks must be positive')
    if a.xattention_threshold is not None:
        if not 0<a.xattention_threshold<=1 or a.mtp=='on':p.error('This experiment requires MTP off and XAttention threshold (0,1]')
    if a.last_dense_tokens is not None and a.last_dense_tokens<0:p.error('Dense tail cannot be negative')
    if a.skip_quality and (a.quality_in_last_timing or a.quality_in_all_timings):p.error('Choose either skipped quality or quality in last timing')
    if min(a.batch,a.repeats,a.quality_tokens)<=0 or a.cache_gb<=0 or (a.draft_kv_blocks is not None and a.draft_kv_blocks<=0):p.error('Positive capacities and counts required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    core=ov.Core();device_name=core.get_property(a.device,'FULL_DEVICE_NAME')
    if 'B60' not in device_name:raise RuntimeError('Selected device is not the B60')
    tokenizer=genai.Tokenizer(a.model)
    cases=[]
    for path in a.prompts:
        obj=json.loads(path.read_text());tokens=obj['cases']['cold-retrieval']['tokens']
        text=tokenizer.decode(tokens,skip_special_tokens=False)
        actual=tokenizer.encode(text,add_special_tokens=False).input_ids.data[0].tolist()
        if tokenizer.decode(actual,skip_special_tokens=False)!=text:
            raise RuntimeError('Retokenization changed decoded input text')
        cases.append((text,actual,obj['expected']))
        write(out/f'input-c{len(cases)}-{len(tokens)}.json',{'original_tokens':len(tokens),'original_token_sha256':digest(tokens),
            'retokenized_tokens':len(actual),'retokenized_token_sha256':digest(actual),
            'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    block_tokens=256 if a.xattention_threshold is not None else 16
    needed=max(len(ids) for _,ids,_ in cases)+(1 if a.skip_quality else a.quality_tokens)
    if a.kv_blocks and a.kv_blocks*block_tokens<needed:raise ValueError('Retokenized input exceeds KV capacity')
    if a.draft_kv_blocks and a.draft_kv_blocks*16<needed:raise ValueError('Retokenized input exceeds draft KV capacity')
    scheduler=genai.SchedulerConfig()
    scheduler.max_num_batched_tokens=a.batch
    scheduler.max_num_seqs=1
    scheduler.enable_prefix_caching=False
    scheduler.cache_size=0 if a.kv_blocks else a.cache_gb
    if a.kv_blocks:scheduler.num_kv_blocks=a.kv_blocks
    if a.xattention_threshold is not None:
        if not 0<a.xattention_threshold<=1:p.error('XAttention threshold must be (0,1]')
        if a.mtp=='on':p.error('This probe keeps the release MTP path separate from XAttention')
        scheduler.use_sparse_attention=True
        scheduler.sparse_attention_config.mode=genai.SparseAttentionMode.XATTENTION
        scheduler.sparse_attention_config.xattention_threshold=a.xattention_threshold
        if a.last_dense_tokens is not None:scheduler.sparse_attention_config.num_last_dense_tokens_in_prefill=a.last_dense_tokens
    props={'KV_CACHE_PRECISION':'u8','DYNAMIC_QUANTIZATION_GROUP_SIZE':a.dq_group,'PERFORMANCE_HINT':'LATENCY'}
    write(out/'config.json',{'openvino':ov.__version__,'genai':genai.__version__,'device':a.device,
        'genai_dll_sha256':hashlib.sha256((Path(genai.__file__).parent/'openvino_genai.dll').read_bytes()).hexdigest(),
        'genai_overlay':a.genai_overlay.name if a.genai_overlay else None,'device_name':device_name,'model':a.model.name,'mtp':a.mtp,'draft_tokens':3 if a.mtp=='on' else 0,
        'scheduler':scheduler.to_string(),'properties':props,'input_mode':'same_decoded_text_retokenized',
        'timing':'TTFT includes prefill and first-token work; throughput is input_tokens / TTFT',
        'cache_prefix':False,'repeats':a.repeats,'draft_kv_blocks':a.draft_kv_blocks,
        'xattention_threshold':a.xattention_threshold,'sparse_attention_config':scheduler.sparse_attention_config.to_string(),
        'cache_budget_scope':'explicit_blocks' if a.kv_blocks else 'main_only' if a.mtp=='off' or a.draft_kv_blocks is not None else 'combined_main_and_default_draft'})
    print('LOAD_START',flush=True);start=time.perf_counter()
    if a.mtp=='on':
        draft_kwargs={}
        if a.draft_kv_blocks is not None:
            draft_scheduler=genai.SchedulerConfig()
            draft_scheduler.max_num_batched_tokens=a.batch
            draft_scheduler.max_num_seqs=1
            draft_scheduler.enable_prefix_caching=False
            draft_scheduler.cache_size=0
            draft_scheduler.num_kv_blocks=a.draft_kv_blocks
            draft_kwargs['scheduler_config']=draft_scheduler
        props['draft_model']=genai.draft_model(a.model,a.device,**draft_kwargs)
    pipe=genai.ContinuousBatchingPipeline(a.model,scheduler,a.device,props)
    write(out/'load.json',{'seconds':time.perf_counter()-start})
    print('LOAD_OK',flush=True)
    def run(text,actual,expected,label,limit):
        cfg=genai.GenerationConfig()
        cfg.max_new_tokens=limit
        cfg.max_length=max(131072,len(actual)+limit+1)
        cfg.do_sample=False
        cfg.apply_chat_template=False
        if a.mtp=='on':cfg.num_assistant_tokens=3
        start=time.perf_counter();first=None
        def stream(part):
            nonlocal first
            if part and first is None:
                first=time.perf_counter()-start
                print('FIRST_TEXT',label,first,flush=True)
            return False
        print('START',label,len(actual),limit,flush=True)
        result=pipe.generate([text],[cfg],stream)[0]
        wall=time.perf_counter()-start;perf=result.perf_metrics
        count=perf.get_num_input_tokens();ttft=perf.get_ttft().mean/1000
        if count!=len(actual):raise RuntimeError(f'Input token count changed: {count} != {len(actual)}')
        output=result.m_generation_ids[0]
        row={'label':label,'input_tokens':count,'wall_seconds':wall,'metric_ttft_seconds':ttft,
            'callback_first_text_seconds':first,'prefill_plus_first_token_tps':count/ttft if ttft>0 else None,
            'effective_tps_to_first_text':count/first if first else None,'input_tokens_per_wall_second':count/wall,
            'generated_tokens':perf.get_num_generated_tokens(),'reported_throughput':perf.get_throughput().mean if perf.get_num_generated_tokens()>1 else None,
            'output':output,'retrieval_correct':sum(code in output for code in expected.values()) if expected and all(isinstance(code,str) for code in expected.values()) else None,
            'quality':score_answer(output,expected),
            'inference_duration_ms':perf.get_inference_duration().mean,
            'raw_token_infer_durations':list(perf.raw_metrics.token_infer_durations),
            'raw_times_to_first_token':list(perf.raw_metrics.m_times_to_first_token)}
        write(out/(label+'.json'),row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['output','raw_token_infer_durations','raw_times_to_first_token']}),flush=True)
        return row
    warm=tokenizer.decode(cases[0][1][:4096],skip_special_tokens=False)
    warm_ids=tokenizer.encode(warm,add_special_tokens=False).input_ids.data[0].tolist()
    run(warm,warm_ids,{},'warmup',1)
    rows=[]
    for repeat in range(a.repeats):
        for index,(text,actual,expected) in enumerate(cases,1):
            limit=a.quality_tokens if (a.quality_in_all_timings or (a.quality_in_last_timing and index==len(cases))) else 1
            rows.append(run(text,actual,expected,f'cold-{len(actual)}-c{index}-r{repeat+1}',limit))
            write(out/'summary.json',rows)
    if not a.skip_quality and not (a.quality_in_last_timing or a.quality_in_all_timings):
        text,actual,expected=cases[-1]
        rows.append(run(text,actual,expected,'cold-retrieval-quality',a.quality_tokens))
        write(out/'summary.json',rows)


if __name__=='__main__':main()

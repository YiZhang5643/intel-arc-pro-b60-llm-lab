"""Quick same-input prefill/decode probe. Warmup is recorded separately."""
import json,random,time,urllib.request
from common import parser,prepare,token_hash
p=parser(__doc__);p.add_argument('--sizes',default='1024,12288');p.add_argument('--tokens',type=int,default=96);a=p.parse_args()
if a.tokens<1:p.error('tokens must be positive')
sizes=[512]+[int(n) for n in a.sizes.split(',')]
if any(n<512 for n in sizes):p.error('sizes must be >=512')
out=prepare(a)
def call(path,data):
    req=urllib.request.Request(a.base_url+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as f:return json.load(f)
rng=random.Random(20261025)
text=''.join(f'Record {i}: station {rng.randrange(100,999)}, quantity {rng.randrange(100000,999999)}, reference {rng.getrandbits(48):012x}.\n' for i in range(12000))
rendered=call('/apply-template',{'messages':[{'role':'user','content':'Use the following ledger as background.\nPROBE_MARKER\nExplain in detail how optimistic version checks prevent lost inventory updates. Provide a numbered example and explain retries and idempotency.'}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt']
pre,post=rendered.split('PROBE_MARKER')
def tok(s,add=False):return call('/tokenize',{'content':s,'add_special':add,'parse_special':True})['tokens']
pre,post,body=tok(pre,True),tok(post),tok(text);rows=[];hashes={}
for i,n in enumerate(sizes):
    tokens=pre+body[:n-len(pre)-len(post)]+post
    assert len(tokens)==n
    hashes[str(n)]=token_hash(tokens)
    data={'prompt':tokens,'n_predict':16 if i==0 else a.tokens,'temperature':0,'seed':20261025,'cache_prompt':False,'stream':True}
    req=urllib.request.Request(a.base_url+'/completion',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter();ttft=None;parts=[];last=None
    print(f'START {n}',flush=True)
    with urllib.request.urlopen(req,timeout=1800) as f:
        for line in f:
            if not line.startswith(b'data:'):continue
            raw=line[5:].strip()
            if raw==b'[DONE]':continue
            frame=json.loads(raw)
            if frame.get('content'):
                if ttft is None:ttft=time.perf_counter()-start
                parts.append(frame['content'])
            if frame.get('timings'):last=frame
    assert last is not None,'No final timings'
    t=last['timings'];assert t['cache_n']==0 and t['prompt_n']==n,t
    row={'test':f'pp{n}','warmup':i==0,'input_tokens':n,'ttft_seconds':ttft,'wall_seconds':time.perf_counter()-start,'timings':t,'output':''.join(parts)}
    rows.append(row)
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'prompt-hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k!='output'}),flush=True)

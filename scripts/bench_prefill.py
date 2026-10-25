import argparse, json, time, random, urllib.request, statistics
from pathlib import Path
from common import parser, prepare, token_hash
p = parser('Exact-token cold prefill')
p.add_argument('--sizes', default='12288')
p.add_argument('--repeats', type=int, default=2)
a = p.parse_args()
if a.repeats < 1:
    p.error('repeats must be positive')

def call(path, data):
    req = urllib.request.Request(a.base_url + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as f:
        return json.load(f)
sizes = [4096] + list(map(int, a.sizes.split(',')))
if any((n < 4096 for n in sizes[1:])) or 4096 in sizes[1:] or len(set(sizes)) != len(sizes):
    p.error('Use unique sizes >4096; 4096 is reserved for warmup')
out = prepare(a)
prompts_path = out / 'prompts.json'
if prompts_path.exists():
    prompts = json.loads(prompts_path.read_text())
else:
    prompts = {}
missing = [n for n in sizes if str(n) not in prompts]
if missing:
    rng = random.Random(20261025)
    text = ''.join((f'Record {i}: sensor {rng.randrange(100, 999)} reports {rng.randrange(100000, 999999)} units; identifier {rng.getrandbits(48):012x}. This is routine inspection data.\n' for i in range(5000)))
    rendered = call('/apply-template', {'messages': [{'role': 'user', 'content': 'Read this data.\nPREFILL_DATA_MARKER\nIgnore the records for the answer. Reply with exactly READY.'}], 'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    pre, post = rendered.split('PREFILL_DATA_MARKER')
    tok = lambda s, add=False: call('/tokenize', {'content': s, 'add_special': add, 'parse_special': True})['tokens']
    pre = tok(pre, True)
    post = tok(post)
    body = tok(text)
    for n in missing:
        prompts[str(n)] = pre + body[:n - len(pre) - len(post)] + post
        assert len(prompts[str(n)]) == n, 'Filler too short for requested size'
    prompts_path.write_text(json.dumps(prompts))
(out / 'prompt-hashes.json').write_text(json.dumps({k: token_hash(v) for k, v in prompts.items()}, indent=2))
rows = []
for n in sizes:
    for rep in range(1 if n == 4096 else a.repeats):
        payload = {'prompt': prompts[str(n)], 'n_predict': 16, 'temperature': 0, 'cache_prompt': False, 'stream': True}
        req = urllib.request.Request(a.base_url + '/completion', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        ttft = None
        frames = []
        timings = {}
        content = ''
        print(f'START {n} run {rep + 1}', flush=True)
        with urllib.request.urlopen(req, timeout=900) as f:
            for line in f:
                if not line.startswith(b'data: '):
                    continue
                line = line[6:].strip()
                if line == b'[DONE]':
                    continue
                frame = json.loads(line)
                frames.append(frame)
                if frame.get('content') and ttft is None:
                    ttft = time.perf_counter() - start
                content += frame.get('content', '')
                if frame.get('timings'):
                    timings = frame['timings']
        wall = time.perf_counter() - start
        assert timings.get('cache_n') == 0, timings
        assert timings.get('prompt_n') == n, timings
        row = {'test': f'pp{n}-r{rep + 1}', 'warmup': n == 4096, 'input_tokens': n, 'ttft_seconds': ttft, 'wall_seconds': wall, 'timings': timings, 'output': content}
        rows.append(row)
        (out / f'{n}-r{rep + 1}-frames.json').write_text(json.dumps(frames, ensure_ascii=False), encoding='utf-8')
        (out / 'prefill-summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
for n in sizes[1:]:
    group = [r for r in rows if r['input_tokens'] == n]
    print('MEAN ' + json.dumps({'tokens': n, 'prefill_tps': statistics.mean((r['timings']['prompt_per_second'] for r in group)), 'ttft': statistics.mean((r['ttft_seconds'] for r in group))}), flush=True)

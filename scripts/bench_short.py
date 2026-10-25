import argparse, json, time, urllib.request
from pathlib import Path
from common import parser, prepare
p = parser('Short code and prose')
a = p.parse_args()
out = prepare(a)

def call(path, data):
    req = urllib.request.Request(a.base_url + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
prompts = {'code': 'Write a complete Python implementation of an LRUCache class with O(1) get and put, using only the standard library. Include a short executable demonstration covering eviction, update, capacity one, and missing keys. Return code only.', 'prose': 'Explain how database transactions prevent lost updates in a small inventory service. Compare optimistic version checks and row-level locking, give a concrete concurrent update example, and explain when retries are needed. Use about 300 words.'}
summary = []
for name, user in prompts.items():
    rendered = call('/apply-template', {'messages': [{'role': 'user', 'content': user}], 'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    data = {'prompt': rendered, 'n_predict': 512, 'temperature': 0.2, 'top_p': 0.9, 'top_k': 20, 'min_p': 0, 'seed': 20261024, 'cache_prompt': False, 'stream': False}
    t = time.monotonic()
    res = call('/completion', data)
    wall = time.monotonic() - t
    (out / (name + '.json')).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    row = {'test': name, 'wall_seconds': wall, 'timings': res.get('timings', {}), 'output': res.get('content', '')}
    summary.append(row)
    (out / 'short-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(row, ensure_ascii=False), flush=True)

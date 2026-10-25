import json, time, urllib.request, re, subprocess, sys, argparse
from pathlib import Path
from common import parser, prepare
p = parser('Small model smoke checks')
p.add_argument('--model',default='nemotron-local')
p.add_argument('--execute-generated-code', action='store_true')
args = p.parse_args()
out = prepare(args)

def chat(name, prompt):
    data = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 1200, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(args.base_url + '/v1/chat/completions', data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        result = json.load(r)
    result['wall_seconds'] = time.monotonic() - t
    (out / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(name, json.dumps(result, ensure_ascii=False), flush=True)
    return result['choices'][0]['message'].get('content', '')
summary = {}
a = chat('arithmetic', 'A warehouse starts with 184 units. It receives 73, ships 96, then receives 28. Of the remaining units, 17 are quarantined and cannot be sold. Return JSON only with keys total_remaining and sellable_remaining, containing integers.')
try:
    obj = json.loads(a.strip())
    summary['arithmetic'] = obj == {'total_remaining': 189, 'sellable_remaining': 172}
except Exception:
    summary['arithmetic'] = False
b = chat('precedence', 'Configuration precedence is command line > environment > file > default. Defaults: timeout=30, retries=2. File: timeout=45, retries=4. Environment: timeout=60. Command line: retries=1. Return JSON only for the final timeout and retries.')
try:
    summary['precedence'] = json.loads(b.strip()) == {'timeout': 60, 'retries': 1}
except Exception:
    summary['precedence'] = False
code = chat('coding', 'Write Python 3 code only, no Markdown, for def compress_ranges(nums). Input is an iterable of integers, possibly unsorted with duplicates. Return a list of (start, end) tuples for maximal consecutive runs, sorted ascending. Empty input returns []. Negative numbers are allowed. Do not mutate the input. Do not import any modules. Example: [3, 1, 2, 2, 7, -2, -1] returns [(-2, -1), (1, 3), (7, 7)].')
if '```' in code:
    blocks = re.findall('```(?:python)?\\s*([\\s\\S]*?)```', code)
    code = blocks[0] if blocks else code
(out / 'generated_code.py').write_text(code, encoding='utf-8')
validator = "\nimport ast,json,sys\nfrom pathlib import Path\ntext=Path(sys.argv[1]).read_text(encoding='utf-8');tree=ast.parse(text)\nfor node in ast.walk(tree):\n if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)):raise ValueError('Forbidden syntax')\n if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Forbidden name')\n if isinstance(node,ast.Attribute) and node.attr.startswith('__'):raise ValueError('Forbidden attribute')\nsafe={k:v for k,v in vars(__builtins__).items() if k in ['len','sorted','set','list','tuple','enumerate','range','min','max','zip','any','all','int','isinstance','iter','next']}\nns={'__builtins__':safe};exec(compile(tree,'model_code','exec'),ns);fn=ns['compress_ranges']\ncases=[([],[]),([1],[(1,1)]),([3,1,2,2,7,-2,-1],[(-2,-1),(1,3),(7,7)]),([5,4,3,2,1],[(1,5)]),([-3,-1,-2,0,2],[(-3,0),(2,2)]),([9,9,9],[(9,9)]),([100,-100],[(-100,-100),(100,100)])]\nfor data,expected in cases:\n original=data.copy();actual=fn(data)\n assert actual==expected,(data,actual,expected)\n assert data==original,'mutated input'\nassert fn(iter([5,3,4]))==[(3,5)]\nprint(json.dumps({'passed':8,'total':8}))\n"
(out / 'validate_code.py').write_text(validator, encoding='utf-8')
if args.execute_generated_code:
    try:
        result = subprocess.run([sys.executable, '-I', str(out / 'validate_code.py'), str(out / 'generated_code.py')], capture_output=True, text=True, timeout=10)
        summary['coding'] = {'passed': result.returncode == 0, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired:
        summary['coding'] = {'passed': False, 'error': 'timeout'}
else:
    summary['coding'] = {'executed': False, 'note': 'Use --execute-generated-code only in an appropriate isolated environment'}
c = chat('chinese', '请用简短中文解释：为什么大模型的预填充速度和逐字生成速度是两个不同的指标？')
summary['chinese_response'] = c
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if not summary['arithmetic'] or not summary['precedence'] or (args.execute_generated_code and not summary['coding'].get('passed')):
    raise SystemExit(1)

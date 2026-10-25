import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
from serve import command
from common import token_hash

class FakeLlama(BaseHTTPRequestHandler):
    bad_cache=False
    previous=[]
    def log_message(self,*args): pass
    def reply(self,obj):
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(obj).encode())
    def do_GET(self):
        self.reply({'total_slots':1,'model_path':'C:/Users/private/model.gguf','default_generation_settings':{'n_ctx':114688,'prompt':'PRIVATE'}})
    def do_POST(self):
        d=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/apply-template':
            text=''.join('['+m['role']+']'+m['content'] for m in d['messages'])+'[assistant]'
            return self.reply({'prompt':text})
        if self.path=='/tokenize':return self.reply({'tokens':([1] if d.get('add_special') else [])+[ord(c)+2 for c in d['content']]})
        if self.path=='/v1/chat/completions':
            text=d['messages'][0]['content']
            content=json.dumps({'total_remaining':189,'sellable_remaining':172}) if 'warehouse' in text else json.dumps({'timeout':60,'retries':1}) if 'precedence' in text else 'def compress_ranges(nums):\n    return []' if 'compress_ranges' in text else '简短说明'
            return self.reply({'choices':[{'message':{'content':content}}]})
        tok=d['prompt']
        if isinstance(tok,str):tok=[ord(c)+2 for c in tok]
        text=''.join(chr(t-2) for t in tok if t>=2)
        codes=dict(re.findall(r'AUTHORITATIVE ENTRY: (PROJECT-[A-Z]+) has access code (ARC-[0-9A-F]+)',text))
        content=json.dumps(codes) if codes else 'READY'
        cache=0
        if d.get('cache_prompt'):
            for x,y in zip(type(self).previous,tok):
                if x!=y:break
                cache+=1
        if type(self).bad_cache:cache=1
        type(self).previous=tok
        t={'cache_n':cache,'prompt_n':len(tok)-cache,'prompt_per_second':1000.0,'predicted_n':3,'predicted_per_second':30.0}
        if d.get('stream'):
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            for obj in [{'content':content,'stop':False},{'content':'','stop':True,'timings':t}]:
                self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode());self.wfile.flush()
        else:self.reply({'content':content,'timings':t})

class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),FakeLlama)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def run_bench(self,script,out,*extra):
        env={**os.environ,'PYTHONUTF8':'1'}
        return subprocess.run([sys.executable,str(R/'scripts'/script),'--label','test','--output',str(out),'--base-url',self.url,*extra],capture_output=True,text=True,encoding='utf-8',timeout=30,env=env)
    def test_prefill_stream_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';res=self.run_bench('bench_prefill.py',out,'--sizes','8192','--repeats','2')
            self.assertEqual(res.returncode,0,res.stderr)
            rows=json.loads((out/'prefill-summary.json').read_text())
            self.assertEqual(len(rows),3)
            self.assertEqual([r['timings']['prompt_n'] for r in rows],[4096,8192,8192])
            self.assertTrue(all(r['ttft_seconds'] is not None and r['ttft_seconds']<=r['wall_seconds'] for r in rows))
            meta=(out/'run-metadata.json').read_text();self.assertNotIn('PRIVATE',meta);self.assertNotIn('private',meta)
            self.assertNotEqual(self.run_bench('bench_prefill.py',out).returncode,0)
    def test_prefill_rejects_cached_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            FakeLlama.bad_cache=True
            try:res=self.run_bench('bench_prefill.py',Path(tmp)/'run','--sizes','8192')
            finally:FakeLlama.bad_cache=False
            self.assertNotEqual(res.returncode,0)
            self.assertIn('AssertionError',res.stderr)
    def test_chat_same_inputs_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashes=[]
            for n in range(2):
                out=Path(tmp)/str(n);res=self.run_bench('bench_chat.py',out,'--size','4096')
                self.assertEqual(res.returncode,0,res.stderr)
                rows=json.loads((out/'summary.json').read_text())
                self.assertEqual(len(rows),4)
                self.assertEqual(rows[0]['retrieval_correct'],3)
                self.assertEqual(rows[1]['retrieval_correct'],3)
                self.assertTrue(all(r['timings']['cache_n']>=3792 for r in rows[1:]))
                hashes.append(json.loads((out/'prompt-hashes.json').read_text()))
            self.assertEqual(*hashes)
    def test_smoke_does_not_execute_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';res=self.run_bench('bench_smoke.py',out)
            self.assertEqual(res.returncode,0,res.stderr)
            result=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(result['arithmetic']);self.assertTrue(result['precedence'])
            self.assertFalse(result['coding']['executed'])
    def test_presets_and_lan_bind(self):
        for path in (R/'presets').glob('*.json'):
            p=json.loads(path.read_text());cmd=command(p,'runtime','models','0.0.0.0',5001,'SYCL0')
            self.assertEqual(cmd[cmd.index('--host')+1],'0.0.0.0')
            self.assertEqual(cmd[cmd.index('--port')+1],'5001')
            self.assertEqual('--spec-type' in cmd,p['mtp'])
        p=json.loads((R/'presets/nemotron-long-chat-mtp.json').read_text());p['ubatch']=p['batch']+1
        with self.assertRaises(ValueError):command(p,'runtime','models','127.0.0.1',5001,'SYCL0')
    def test_hash_and_published_evidence(self):
        self.assertEqual(token_hash([1,2]),hashlib.sha256(b'[1,2]').hexdigest())
        res=subprocess.run([sys.executable,str(R/'scripts/check_evidence.py')],capture_output=True,text=True)
        self.assertEqual(res.returncode,0,res.stderr)

if __name__=='__main__':unittest.main()

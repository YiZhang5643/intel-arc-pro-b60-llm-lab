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
from serve import command, environment
from inspect_gguf import inspect
import struct
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
    def test_prepare_prompts_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            FakeLlama.previous=[987654321]
            out=Path(tmp)/'run'
            result=self.run_bench('bench_chat.py',out,'--size','4096','--prepare-only')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((out/'prompt-hashes.json').exists())
            self.assertFalse((out/'summary.json').exists())
            self.assertEqual(FakeLlama.previous,[987654321])
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

    def test_embedded_mtp_uses_main_weights(self):
        p=json.loads((R/'presets/qwen38-mtp3.json').read_text())
        cmd=command(p,'runtime','models','0.0.0.0',5001,'SYCL0')
        self.assertIn('--spec-type',cmd);self.assertNotIn('-md',cmd)
        self.assertEqual(cmd[cmd.index('--alias')+1],'qwen38-27b-local')
        self.assertEqual(environment(p)['GGML_SYCL_ENABLE_GRAPH'],'0')
        self.assertEqual(environment(p)['GGML_SYCL_GDN_BLOCKED'],'0')
        self.assertEqual(environment(json.loads((R/'presets/qwen38-recommended.json').read_text()))['GGML_SYCL_GDN_BLOCKED'],'1')
        p['sycl_graph']=True
        self.assertEqual(environment(p)['GGML_SYCL_ENABLE_GRAPH'],'1')
        p['mtp_mode']='unknown'
        with self.assertRaises(ValueError):command(p,'runtime','models','0.0.0.0',5001,'SYCL0')
    def test_gguf_inspection_skips_arrays_and_reads_tensors(self):
        def text(s):
            b=s.encode();return struct.pack('<Q',len(b))+b
        metadata=text('general.architecture')+struct.pack('<I',8)+text('qwen35')
        metadata+=text('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,2)+text('a')+text('b')
        metadata+=text('qwen35.nextn_predict_layers')+struct.pack('<II',4,1)
        tensor=text('blk.64.nextn.eh_proj.weight')+struct.pack('<IQQIQ',2,32,64,21,0)
        raw=b'GGUF'+struct.pack('<IQQ',3,1,3)+metadata+tensor
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.gguf';p.write_bytes(raw);result=inspect(p)
            self.assertEqual(result['metadata']['general.architecture'],'qwen35')
            self.assertNotIn('tokenizer.ggml.tokens',result['metadata'])
            self.assertEqual(result['tensor_type_counts'],{'21':1})
            self.assertEqual(result['parameter_counts_by_type'],{'21':2048})
            self.assertEqual(len(result['nextn_tensors']),1)


    def test_installer_preserves_user_files_and_forwarder_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            deployment=Path(tmp)/'deployment with spaces';(deployment/'models').mkdir(parents=True)
            (deployment/'Start-Qwen.cmd').write_text('original cmd')
            (deployment/'qwen26b_server.py').write_text('original python')
            result=subprocess.run([sys.executable,str(R/'scripts/install_qwen_launcher.py'),'--deployment-dir',str(deployment)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(next(deployment.glob('Start-Qwen.before-tuning-*.cmd')).read_text(),'original cmd')
            self.assertEqual(next(deployment.glob('qwen26b_server.before-tuning-*.py')).read_text(),'original python')
            result=subprocess.run([sys.executable,str(deployment/'qwen26b_server.py'),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--draft-ubatch',result.stdout)
    @unittest.skipUnless(os.name=='nt','Windows process handles')
    def test_stop_already_exited_recorded_child(self):
        from manage_server import stop
        with tempfile.TemporaryDirectory() as tmp:
            child=subprocess.Popen([sys.executable,'-c','pass'],creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait(timeout=10)
            record=Path(tmp)/'server.json'
            record.write_text(json.dumps({'pid':child.pid,'exe':sys.executable}))
            stop(record)

    @unittest.skipUnless(os.name=='nt','Windows process handles')
    def test_stop_waits_for_query_denied_terminating_child(self):
        import ctypes
        from types import SimpleNamespace
        from unittest.mock import patch
        from manage_server import stop
        real=ctypes.WinDLL('kernel32',use_last_error=True)
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
        timer=None
        def denied_query(*args):
            nonlocal timer
            timer=threading.Timer(0.1,child.terminate)
            timer.start()
            ctypes.set_last_error(5)
            return 0
        proxy=SimpleNamespace(**{name:getattr(real,name) for name in ['OpenProcess','TerminateProcess','WaitForSingleObject','CloseHandle']},QueryFullProcessImageNameW=denied_query)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                record=Path(tmp)/'server.json'
                record.write_text(json.dumps({'pid':child.pid,'exe':sys.executable}))
                with patch('ctypes.WinDLL',return_value=proxy):stop(record)
                self.assertIsNotNone(child.poll())
        finally:
            if timer:timer.join()
            if child.poll() is None:child.terminate()
            child.wait(timeout=10)

    def test_snapshot_redacts_paths_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'input'/'probe';run.mkdir(parents=True)
            (run/'summary.json').write_text(json.dumps([{'test':'cold','timings':{'cache_n':0}}]))
            (run/'actual-launch.local.json').write_text(json.dumps({'args':['C:\\Users\\private\\llama-server.exe','-m','C:\\models\\model.gguf'],'environment_overrides':{}}))
            out=root/'snapshot'
            result=subprocess.run([sys.executable,str(R/'scripts/record_snapshot.py'),'--input-root',str(root/'input'),'--output',str(out),'--labels','probe'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('private',(out/'probe/config.json').read_text())
            result=subprocess.run([sys.executable,str(R/'scripts/check_evidence.py'),'--directory',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            (out/'probe/summary.json').write_text('tampered')
            result=subprocess.run([sys.executable,str(R/'scripts/check_evidence.py'),'--directory',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

    def test_hash_and_published_evidence(self):
        self.assertEqual(token_hash([1,2]),hashlib.sha256(b'[1,2]').hexdigest())
        res=subprocess.run([sys.executable,str(R/'scripts/check_evidence.py')],capture_output=True,text=True)
        self.assertEqual(res.returncode,0,res.stderr)

if __name__=='__main__':unittest.main()

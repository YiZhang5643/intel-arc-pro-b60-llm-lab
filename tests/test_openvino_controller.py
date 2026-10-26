import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.model=self.root/'model';self.model.mkdir()
        self.prompts=self.root/'prompts.json'
        self.prompts.write_text(json.dumps({'cases':{'cold-retrieval':{'tokens':list(range(1024))}}}))
        self.preset=self.root/'preset.json'
        self.preset.write_text(json.dumps({'backend':'openvino-genai','mtp':'off','batch':2048,'kv_blocks':8,'xattention_threshold':0.1}))

    def run_cli(self,*extra):
        return subprocess.run([sys.executable,str(ROOT/'scripts/run_openvino_bench.py'),
            '--deployment-dir',str(self.root),'--model',str(self.model),'--python',sys.executable,
            '--prompts',str(self.prompts),'--output',str(self.root/'output'),
            '--preset',str(self.preset),'--dry-run',*extra],capture_output=True,text=True)

    def test_explicit_override_and_no_output_mutation(self):
        result=self.run_cli('--batch','4096')
        self.assertEqual(result.returncode,0,result.stderr)
        config=json.loads(result.stdout)
        self.assertEqual(config['batch'],4096)
        self.assertEqual(config['mtp'],'off')
        self.assertFalse((self.root/'output').exists())

    def test_invalid_mtp_sparse_combination(self):
        result=self.run_cli('--mtp','on')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('MTP off',result.stderr)
        self.assertFalse((self.root/'output').exists())

    def test_full_output_budget_in_capacity_check(self):
        result=self.run_cli('--kv-blocks','5','--quality-tokens','512')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('KV capacity',result.stderr)

    def test_corrected_preset_rejects_unpatched_runtime(self):
        preset=json.loads(self.preset.read_text())
        preset['requires_genai_patch_sha256']='a'*64
        self.preset.write_text(json.dumps(preset))
        result=self.run_cli()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('corrected GenAI overlay',result.stderr)
        self.assertFalse((self.root/'output').exists())

    def test_snapshot_mismatch(self):
        preset=json.loads(self.preset.read_text())
        preset.update(model_repository='one/model',model_revision='a'*40)
        self.preset.write_text(json.dumps(preset))
        (self.model/'snapshot-manifest.json').write_text(json.dumps({'repo':'other/model','revision':'a'*40}))
        result=self.run_cli()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('does not match',result.stderr)


if __name__=='__main__':unittest.main()

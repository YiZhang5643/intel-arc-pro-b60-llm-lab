"""A timed-out venv launcher must not leave a GPU-owning grandchild alive."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gpu_probe import run_process
from manage_server import stop


@unittest.skipUnless(os.name=='nt','Windows process tree regression')
class ProbeProcessTests(unittest.TestCase):
    def test_timeout_stops_grandchild(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'child.json'
            code="import subprocess,sys,json,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid,'exe':sys.executable})); time.sleep(60)"
            try:
                with open(os.devnull,'wb') as log:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        run_process([sys.executable,'-c',code,str(path)],log,os.environ.copy(),2)
                self.assertTrue(path.exists(),'Grandchild was not started')
                pid=json.loads(path.read_text())['pid']
                handle=kernel.OpenProcess(0x100000,False,pid)
                if handle:
                    try:self.assertEqual(kernel.WaitForSingleObject(handle,0),0,'Grandchild survived timeout')
                    finally:kernel.CloseHandle(handle)
                else:self.assertEqual(ctypes.get_last_error(),87)
            finally:
                if path.exists():stop(path)


if __name__=='__main__':unittest.main()

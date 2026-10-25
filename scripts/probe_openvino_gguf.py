"""Report installed OpenVINO capabilities and try the given GGUF on an explicit GPU."""
import argparse
import json
import time
import openvino as ov
import openvino_genai as genai
from openvino.frontend import FrontEndManager

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True)
p.add_argument('--device',default='GPU.1')
a=p.parse_args()
core=ov.Core()
print(json.dumps({'openvino':ov.__version__,'genai':genai.__version__,
    'frontends':FrontEndManager().get_available_front_ends(),
    'devices':{d:core.get_property(d,'FULL_DEVICE_NAME') for d in core.available_devices}},indent=2),flush=True)
if 'B60' not in core.get_property(a.device,'FULL_DEVICE_NAME'):
    raise RuntimeError('Selected device is not the B60')
start=time.perf_counter()
try:
    pipe=genai.LLMPipeline(a.model,a.device)
    print('LOAD_OK seconds=',time.perf_counter()-start,flush=True)
    print(pipe.generate('The capital of France is',max_new_tokens=12),flush=True)
except Exception as e:
    print(json.dumps({'stage':'pipeline_load_or_short_generation','seconds':time.perf_counter()-start,
        'error_type':type(e).__name__,'error':str(e)},indent=2),flush=True)
    raise SystemExit(2)

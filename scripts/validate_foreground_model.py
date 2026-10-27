"""Real checkpoint smoke test with a synthetic RGB/alpha input, not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sammie.foreground.config import ForegroundSettings
from sammie.foreground.model import CorridorModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',choices=['blue','green'],default='green')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--output',default='.test-output/corridorkey-model-smoke.json')
    a=p.parse_args()
    torch.set_num_threads(6)
    options=ForegroundSettings(screen=a.screen,device=a.device,resolution=a.resolution,precision='fp32')
    rgb=np.zeros((64,96,3),np.float32)
    rgb[...,2 if a.screen=='blue' else 1]=.4
    rgb[12:54,30:66]=[.5,.1,.07]
    alpha=np.zeros((64,96),np.float32)
    alpha[12:54,30:66]=1.
    original=alpha.copy()
    start=time.perf_counter()
    model=CorridorModel(options,print)
    load=time.perf_counter()-start
    start=time.perf_counter()
    output=model.predict(rgb,alpha)
    seconds=time.perf_counter()-start
    np.testing.assert_array_equal(alpha,original)
    assert output.shape==(a.resolution,a.resolution,3) and np.isfinite(output).all()
    assert not model.engine.model.training
    assert all(p.grad is None for p in model.engine.model.parameters())
    report=dict(passed=True,screen=a.screen,device=a.device,resolution=a.resolution,
        torch=torch.__version__,load_seconds=load,inference_seconds=seconds,
        output_dtype=str(output.dtype),output_range=[float(output.min()),float(output.max())],
        evaluation_mode=True,no_gradients=True,input_alpha_unchanged=True,
        qualification='Synthetic API/loading smoke test; not foreground accuracy or temporal validation.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

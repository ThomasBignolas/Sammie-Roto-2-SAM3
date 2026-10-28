"""Read-only native-resolution preview timing, including Qt transfer and heartbeat.

Run: python -m scripts.benchmark_preview --root path/to/shot.sammie-data --frames 0 1
Does not load segmentation/matting models, write the project or alter settings.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.preview import PreviewController
from sammie.projects import bind_root
from sammie.settings_manager import initialize_settings


def benchmark(root, frames, repeats=6):
    if not frames or min(frames) < 0 or repeats < 1:
        raise ValueError('Supply nonnegative frame indices and at least one repeat.')
    app = QApplication.instance() or QApplication([])
    manager = initialize_settings(str(root))
    manager.load_session_settings()
    bind_root(root, manager)
    points = manager.load_points() or []
    image = core.load_base_frame(frames[0])
    core.VideoInfo.height, core.VideoInfo.width = image.shape[:2]
    core.VideoInfo.total_frames = max(frames)+1
    del image
    controller = PreviewController()
    responses, failures, ticks = [], [], []
    controller.ready.connect(lambda frame, pixmap: responses.append((frame, pixmap.size().toTuple())))
    controller.failed.connect(failures.append)
    timer=QTimer();timer.timeout.connect(lambda: ticks.append(time.perf_counter()));timer.start(10)
    report=dict(resolution=[core.VideoInfo.width,core.VideoInfo.height], modes={})
    try:
        for mode in ('None','Segmentation-Edit','Segmentation-Matte','Segmentation-BGcolor','Matting-Matte'):
            if mode == 'Matting-Matte' and not any((Path(root)/'matting'/f'{f:05d}').is_dir() for f in frames):
                report['modes'][mode] = {'skipped': 'No matting outputs for these frames.'}
                continue
            view=dict(view_mode=mode,antialias=False,show_masks=True,show_outlines=True,background_style='checkerboard')
            timings=[];hits=[];ticks.clear()
            stage_start = time.perf_counter()
            for index in range(repeats):
                count=len(responses);begin=time.perf_counter()
                controller.request(frames[index%len(frames)],view,points,manager)
                while len(responses)==count and not failures and time.perf_counter()-begin<120:
                    app.processEvents();time.sleep(.001)
                if failures: raise RuntimeError(failures[-1])
                if len(responses)==count: raise TimeoutError('Preview did not respond.')
                timings.append(time.perf_counter()-begin);hits.append(controller.last_cache_hit)
            ticks_with_bounds = [stage_start, *ticks, time.perf_counter()]
            report['modes'][mode]=dict(seconds=timings,cache_hits=hits,
                median_seconds=statistics.median(timings),cache_bytes=controller.cache_bytes,
                largest_heartbeat_gap_seconds=max((b-a for a,b in zip(ticks_with_bounds,ticks_with_bounds[1:])),default=0))
            try:
                import psutil
                process=psutil.Process(controller.process.processId())
                report['worker_rss_mib']=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)])/1024**2
            except (ImportError, OSError): pass
    finally:
        timer.stop();controller.close()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[0,1])
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report=benchmark(args.root,args.frames,args.repeats)
    text=json.dumps(report,indent=2)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()

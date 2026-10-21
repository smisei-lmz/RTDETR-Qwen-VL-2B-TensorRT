"""Create deterministic PPM fixtures or extract video into RGB PPM files."""
import argparse
import json
from pathlib import Path


def fixtures(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for frame,(w,h) in enumerate([(320,180),(641,359),(96,160),(321,241),(640,360),(65,65)]):
        x0,y0,x1,y1=w//5,h//4,w//2,h//2
        pixels=bytearray(w*h*3)
        for y in range(y0,y1):
            for x in range(x0,x1): pixels[(y*w+x)*3]=255
        path=directory/f'{frame:05}.ppm'
        path.write_bytes(f'P6\n{w} {h}\n255\n'.encode()+pixels)
        records.append(dict(file=path.name,width=w,height=h,expected_box=[x0,y0,x1,y1]))
    (directory/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return records


def video(directory, source=None, frames=120):
    import cv2
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if any(directory.glob('*.ppm')):
        raise RuntimeError('Video output already contains PPM frames; choose a fresh --out directory to avoid mixing recordings')
    if source is None:
        source=directory.parent/'teaching.avi'
        writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,360))
        if not writer.isOpened(): raise RuntimeError('MJPG video writer unavailable; supply --video')
        for i in range(frames):
            frame=np.zeros((360,640,3),dtype=np.uint8);x=(i*5)%500
            cv2.rectangle(frame,(x,100),(x+100,200),(0,0,255),-1);writer.write(frame)
        writer.release()
    cap=cv2.VideoCapture(str(source));count=0
    if not cap.isOpened(): raise RuntimeError(f'Cannot read video {source}')
    try:
        while count<frames:
            ok,frame=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]
            (directory/f'{count:05}.ppm').write_bytes(f'P6\n{w} {h}\n255\n'.encode()+rgb.tobytes())
            count+=1
    finally:cap.release()
    if not count:raise RuntimeError('Video contains no readable frames')
    (directory/'video.json').write_text(json.dumps(dict(source=str(source),frames=count,
      timestamp_semantics='Replayed at configured rate; no original PTS preservation'),indent=2),encoding='utf-8')
    return count


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['fixtures','video']);p.add_argument('--out',required=True)
    p.add_argument('--video');p.add_argument('--frames',type=int,default=120);a=p.parse_args()
    print(fixtures(a.out) if a.task=='fixtures' else video(a.out,a.video,a.frames))

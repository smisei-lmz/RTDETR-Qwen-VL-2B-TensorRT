"""Concrete daily commands. Shared source is intentional: one evolving project."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def run(command,cwd=ROOT,out=None,expected=(0,),env=None,timeout=None):
    command=[str(value) for value in command]
    print('+ '+shlex.join(command),flush=True)
    effective=os.environ.copy()
    if env:effective.update(env)
    if out:
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        with Path(out).open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=cwd,env=effective,stdout=stream,stderr=subprocess.PIPE,
                                  text=True,encoding='utf-8',errors='replace',timeout=timeout)
    else:
        result=subprocess.run(command,cwd=cwd,env=effective,timeout=timeout)
    if out and result.stderr:
        print(result.stderr.strip())
        Path(str(out)+'.stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode not in expected:
        raise RuntimeError(f'Command exited {result.returncode}. Install/fix its prerequisite; no mock substitution was made.')
    return result


def command_info(command):
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
        return dict(exit=result.returncode,output=(result.stdout+result.stderr).strip())
    except (FileNotFoundError,subprocess.TimeoutExpired) as exc:return dict(error=str(exc))


def environment(path):
    report=dict(os=platform.platform(),python=sys.version,executable=PY,
       packages={name:importlib.util.find_spec(name) is not None for name in ['torch','rclpy','cv2','numpy','onnxruntime','transformers','tensorrt']},
       commands={name:command_info(command) for name,command in {
           'gpu':['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
           'compiler':['g++','--version'],'cmake':['cmake','--version'],'cuda':['nvcc','--version']}.items()})
    release=Path('/etc/nv_tegra_release')
    if release.is_file():report['jetson_release']=release.read_text()
    save(path,report);print(json.dumps(report,ensure_ascii=False,indent=2));return report


def require_file(path):
    path=Path(path)
    if not path.is_file():raise RuntimeError(f'Missing {path}; run the prerequisite day first')
    return path


def absolute(value):
    p=Path(value).expanduser();return p if p.is_absolute() else ROOT/p


def cpp_exe(kind='cpu',name='edge_pipeline'):
    directory=ROOT/'build'/kind
    candidates=[directory/(name+('.exe' if os.name=='nt' else '')),directory/'Release'/(name+'.exe')]
    for candidate in candidates:
        if candidate.is_file():return candidate
    raise RuntimeError(f'Build {kind} first: executable {name} is missing')


def build_cpp(kind,cfg,jetson=False,shared=False,trt=False):
    if not shutil.which('cmake'):raise RuntimeError('Install CMake and a C++17 compiler in Ubuntu first')
    cuda=kind!='cpu'
    args=['cmake','-S',ROOT/('jetson' if jetson else 'pc'),'-B',ROOT/'build'/kind,
          '-DCMAKE_BUILD_TYPE=RelWithDebInfo',f'-DEDGE_WITH_CUDA={"ON" if cuda else "OFF"}',
          f'-DEDGE_WITH_TENSORRT={"ON" if trt else "OFF"}']
    if cuda:args += [f'-DCMAKE_CUDA_ARCHITECTURES={cfg["jetson_cuda_arch" if jetson else "cuda_arch"]}']
    if jetson:args += [f'-DEDGE_WITH_JETSON={"ON" if shared else "OFF"}']
    run(args);run(['cmake','--build',ROOT/'build'/kind,'--config','RelWithDebInfo','-j','4'])


def cpp_run(folder,kind='cpu',backend='cpu',frames=120,fps=60,extra=(),name='run'):
    events=folder/f'{name}.jsonl';summary=folder/f'{name}.json'
    run([cpp_exe(kind),'--backend',backend,'--frames',frames,'--fps',fps,
         '--summary',summary,*extra],out=events,timeout=max(60,frames/max(1,fps)*3+60))
    data=json.loads(summary.read_text());assert data['all_slots_free']
    assert data['generated']==data['completed']+data['drop_oldest']+data['drop_no_slot']
    return events,data


def model(task,cfg,*extra,board=False):
    command=[PY,ROOT/'tools/model_tools.py',task,'--weights',cfg['weights']]
    if cfg['image']:command += ['--image',absolute(cfg['image'])]
    command += list(extra)
    env={'EDGE_MODEL_DIR':str(ROOT/'artifacts/jetson-models')} if board else None
    return run(command,env=env)


def ros_shell(command,overlay=True):
    if os.name!='posix':raise RuntimeError('ROS days run INSIDE WSL Ubuntu 22.04, not native PowerShell')
    setup=Path('/opt/ros/humble/setup.bash')
    require_file(setup)
    prefix='source '+shlex.quote(str(setup))
    if overlay:prefix+=' && source '+shlex.quote(str(require_file(ROOT/'ros_ws/install/setup.bash')))
    return ['bash','-c',prefix+' && exec '+shlex.join([str(x) for x in command])]


def ros_build():
    run(ros_shell([PY,'-c','from colcon_core.command import main; raise SystemExit(main())','build',
        '--symlink-install','--packages-select','edge_interfaces','edge_assistant'],overlay=False),cwd=ROOT/'ros_ws')


def ros_launch(cfg,backend='cpu',vlm='mock',auto=False,board=False,delay=2,inspection=False):
    kind='jetson' if board else ('trt' if backend=='trt' else ('cuda' if backend=='cuda' else 'cpu'))
    command=['ros2','launch','edge_assistant','course.launch.py',
      'pipeline_exe:='+str(cpp_exe(kind,'edge_camera' if board else 'edge_pipeline')),
      'producer_kind:='+('jetson' if board else 'pc'),'backend:='+backend,'vlm_backend:='+vlm,
      'auto_ask:='+str(auto).lower(),'mock_delay_s:='+str(delay),'model:='+cfg['vlm_model']]
    if backend=='trt':
        engine=absolute(cfg['jetson_engine' if board else 'engine']);require_file(engine)
        command += ['engine:='+str(engine)]
        names=ROOT/'artifacts'/('jetson-models' if board else 'models')/'reference.json'
        if names.is_file():command += ['names_json:='+str(names)]
        if not board:command += ['source:='+str(ROOT/'data/reference')]
    if board:command += ['camera:='+cfg['camera'],'camera_format:='+cfg['camera_format'],'matrix:='+str(cfg['matrix']),
        'camera_width:='+str(cfg['camera_width']),'camera_height:='+str(cfg['camera_height']),'full_range:='+str(cfg['full_range']).lower()]
    if vlm!='mock':command += ['timeout_s:=60.0','max_answer_age_s:=90.0']
    if inspection:
        command = [arg for arg in command if not arg.startswith(('source:=','auto_ask:='))]
        command += ['auto_ask:=false','inspection_enabled:=true',
                    'occupied_s:='+str(cfg['occupied_s']),'absent_s:='+str(cfg['absent_s']),
                    'workstation_roi:='+cfg['workstation_roi'],'report_dir:='+str(absolute(cfg['report_dir']))]
        if not board:
            source = absolute(cfg['inspection_source'])
            if not source.is_dir() or not any(source.glob('*.ppm')):
                raise RuntimeError('Inspection needs a workstation PPM sequence. Run Day19 --video YOUR_VIDEO --frames 900 first.')
            command += ['source:='+str(source)]
    return ros_shell(command)


def ros_experiment(folder,cfg,observer,backend='cpu',auto=False,delay=2):
    command=ros_launch(cfg,backend=backend,auto=auto,delay=delay)
    with (folder/'launch.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            run(ros_shell([PY,ROOT/'tools'/observer[0],*observer[1:]]),out=folder/'observation.json',timeout=70)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise RuntimeError('ROS launch failed to stop gracefully; inspect launch.log')


def report(folder,board=False):
    rows=[]
    for path in sorted((ROOT/'runs').glob('day*/*.json')):
        try:data=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):continue
        if isinstance(data,dict) and 'completed' in data:
            rows.append(f'| {path.parent.name}/{path.name} | {data["completed"]} | {data.get("drop_oldest",0)} | {data.get("latency_p99_ms","—")} |')
    content='# 实际运行记录汇总\n\n表中只汇总已经运行生成的文件，不填入预测成绩。\n\n'
    content+='| 实验 | 完成帧数 | 丢旧帧 | p99 ms |\n|---|---:|---:|---:|\n'+'\n'.join(rows)
    content+='\n\n## 仍需核对\n\n- 模型/数据集/精度门限是否固定。\n- CUDA stage 与端到端耗时是否分开。\n- ROS/VLM 开关对检测尾延迟的影响。\n- Jetson 模式、相机格式、颜色范围与共享能力实测。\n'
    content+='\n板端阶段已进入。\n' if board else '\n此报告是 PC 阶段，不能作为 Jetson 性能证据。\n'
    (folder/'report.md').write_text(content,encoding='utf-8');print(folder/'report.md')


def run_day(day):
    parser=argparse.ArgumentParser(description=f'Day {day:02d}: see this directory README')
    for key in ['weights','image','video','dataset','engine','vlm_model','camera','camera_format','inspection_source','workstation_roi','report_dir']:
        parser.add_argument('--'+key.replace('_','-'))
    parser.add_argument('--frames',type=int);parser.add_argument('--seconds',type=int)
    parser.add_argument('--profile',action='store_true');parser.add_argument('--matrix',type=int)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.example.json').read_text(encoding='utf-8'))
    if (ROOT/'config.json').is_file():cfg.update(json.loads((ROOT/'config.json').read_text(encoding='utf-8')))
    for key in cfg:
        value=getattr(args,key,None)
        if value is not None:cfg[key]=value
    folder=ROOT/'runs'/f'day{day:02d}';folder.mkdir(parents=True,exist_ok=True)
    save(folder/'invocation.json',dict(day=day,config=cfg,python=PY,arguments=sys.argv[1:]))
    frames=lambda default:args.frames if args.frames is not None else default
    if day==1:environment(folder/'environment.json')
    elif day==2:run([PY,ROOT/'tools/assets.py','fixtures','--out',ROOT/'data/fixtures'])
    elif day==3:model('reference',cfg)
    elif day==4:model('onnx',cfg)
    elif day==5:model('engine',cfg)
    elif day==6:
        model('compare',cfg,'--report',folder/'parity.json');model('evaluate',cfg,'--dataset',cfg['dataset'])
    elif day==7:
        build_cpp('cpu',cfg);run([cpp_exe('cpu','core_tests')]);cpp_run(folder,frames=frames(30),extra=['--serial'])
    elif day==8:cpp_run(folder,frames=frames(180),fps=60,extra=['--lanes','1','--work-ms','15'])
    elif day==9:
        run([cpp_exe('cpu','core_tests')])
        for pool in [3,8]:cpp_run(folder,frames=frames(120),fps=240,extra=['--pool',str(pool),'--work-ms','40'],name=f'pool{pool}')
    elif day==10:
        for size in [1,2,8]:cpp_run(folder,frames=frames(180),fps=120,extra=['--queue',str(size),'--work-ms','60'],name=f'queue{size}')
    elif day==11:
        result=run([cpp_exe('cpu'),'--fault','early-recycle','--frames','20'],out=folder/'fault.jsonl',expected=(2,))
        if 'stale/early-recycled lease' not in result.stderr:raise RuntimeError('Expected ownership failure was not observed')
        for i in range(10):cpp_run(folder,frames=12,fps=240,name=f'restart{i:02d}')
    elif day==12:
        build_cpp('cuda',cfg);cpp_run(folder,'cuda','cuda',frames(30),extra=['--serial'])
    elif day==13:cpp_run(folder,'cuda','cuda',frames(18),extra=['--source',ROOT/'data/fixtures','--verify-preprocess'])
    elif day==14:
        for lanes in [1,2]:cpp_run(folder,'cuda','cuda',frames(240),fps=120,extra=['--lanes',str(lanes)],name=f'lanes{lanes}')
    elif day==15:
        require_file(absolute(cfg['engine']));build_cpp('trt',cfg,trt=True)
        cpp_run(folder,'trt','trt',frames(30),extra=['--serial','--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference'])
    elif day==16:cpp_run(folder,'trt','trt',frames(180),fps=60,extra=['--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference','--lanes','2'])
    elif day==17:
        events,_=cpp_run(folder,'trt','trt',1,extra=['--serial','--verify-preprocess','--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference'])
        model('compare',cfg,'--events',events,'--report',folder/'cpp-parity.json')
    elif day==18:
        events,_=cpp_run(folder,'trt','trt',frames(300),fps=120,extra=['--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference'])
        run([PY,ROOT/'tools/analyze.py',events,'--out',folder/'latency.json'])
        if args.profile:
            run(['nsys','profile','--trace=cuda,nvtx,osrt','--output',folder/'timeline',cpp_exe('trt'),
                 '--backend','trt','--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference','--frames','120'],out=folder/'profile-events.jsonl')
    elif day==19:
        source=absolute(cfg['inspection_source'])
        command=[PY,ROOT/'tools/assets.py','video','--out',source,'--frames',frames(120)]
        if cfg['video']:command += ['--video',absolute(cfg['video'])]
        run(command);cpp_run(folder,'trt','trt',frames(120),fps=30,extra=['--engine',absolute(cfg['engine']),'--source',source])
    elif day==20:ros_build();run(ros_shell([PY,ROOT/'tools/ros_basics.py']))
    elif day==21:ros_experiment(folder,cfg,['ros_observe.py','--seconds','8'])
    elif day==22:ros_experiment(folder,cfg,['ros_observe.py','--seconds','8','--snapshots'],backend='cuda')
    elif day==23:ros_experiment(folder,cfg,['ros_observe.py','--seconds','13','--snapshots','--answers'],auto=True)
    elif day==24:
        image=absolute(cfg['image']) if cfg['image'] else ROOT/'artifacts/models/vlm-input.jpg'
        run([PY,ROOT/'tools/ask_image.py',require_file(image),'--backend','qwen2_vl','--model',cfg['vlm_model']],out=folder/'answer.json')
    elif day==25:run(ros_launch(cfg,backend='trt',vlm='qwen2_vl',inspection=True))
    elif day==26:ros_experiment(folder,cfg,['ros_action_test.py'],auto=False,delay=2)
    elif day==27:
        run([PY,'-m','unittest','discover','-s',ROOT/'tests','-v']);run([PY,'-m','unittest','discover','-s',ROOT/'ros_ws/tests','-v'])
        run([cpp_exe('cpu','core_tests')]);cpp_run(folder,frames=60,fps=120,extra=['--work-ms','40'])
    elif day==28:
        seconds=args.seconds or 1800;events,_=cpp_run(folder,'trt','trt',frames(seconds*30),fps=30,
            extra=['--engine',absolute(cfg['engine']),'--source',ROOT/'data/reference'])
        run([PY,ROOT/'tools/analyze.py',events,'--warmup','200','--out',folder/'soak.json'])
    elif day==29:report(folder)
    elif day==30:run(ros_launch(cfg,backend='trt',vlm='qwen2_vl',inspection=True))
    elif 31<=day<=37:
        require_file('/etc/nv_tegra_release')
        if day==31:
            environment(folder/'environment.json');run(['v4l2-ctl','-d',cfg['camera'],'--all','--list-formats-ext'],out=folder/'camera.txt')
        elif day==32:
            build_cpp('jetson',cfg,jetson=True)
            camera_run(folder,cfg,frames(120),mode='copy',backend='cuda')
        elif day==33:
            result=run([cpp_exe('jetson','capture_probe'),'--camera',cfg['camera'],'--format',cfg['camera_format'],
              '--width',cfg['camera_width'],'--height',cfg['camera_height'],'--export','--frames','60'],out=folder/'export.jsonl',expected=(0,2))
            if result.returncode and 'EXPBUF' not in result.stderr:raise RuntimeError('Capture failed for a reason other than EXPBUF support')
            save(folder/'capability.json',dict(export_supported=result.returncode==0,details=result.stderr,
                note='Export support does not prove CUDA import; NVIDIA allocation + V4L2 import is tested on Day34'))
        elif day==34:
            build_cpp('jetson',cfg,jetson=True,shared=True)
            camera_run(folder,cfg,frames(120),mode='shared',backend='cuda',extra=['--verify-source'])
        elif day==35:
            model('reference',cfg,board=True);model('engine',cfg,board=True);model('compare',cfg,board=True)
            build_cpp('jetson',cfg,jetson=True,shared=True,trt=True)
            camera_run(folder,cfg,frames(300),mode='shared',backend='trt')
        elif day==36:ros_build();run(ros_launch(cfg,backend='trt',vlm='qwen2_vl',board=True,inspection=True))
        elif day==37:
            camera_run(folder,cfg,2147483647,mode='shared',backend='trt',extra=['--duration-s',str(args.seconds or 1800)])
            run([PY,ROOT/'tools/analyze.py',folder/'camera.jsonl','--warmup','100','--out',folder/'latency.json']);report(folder,board=True)
    else:raise ValueError('day must be 1..37')


def camera_run(folder,cfg,frames,mode,backend,extra=()):
    command=[cpp_exe('jetson','edge_camera'),'--camera',cfg['camera'],'--format',cfg['camera_format'],
       '--width',cfg['camera_width'],'--height',cfg['camera_height'],'--mode',mode,'--backend',backend,
       '--matrix',cfg['matrix'],'--frames',frames,*extra]
    if cfg['full_range']:command += ['--full-range']
    if backend=='trt':command += ['--engine',require_file(absolute(cfg['jetson_engine']))]
    result=run(command,out=folder/'camera.jsonl')
    for line in reversed(result.stderr.splitlines()):
        try:summary=json.loads(line)
        except ValueError:continue
        if isinstance(summary,dict) and 'completed' in summary:
            save(folder/'camera-summary.json',summary);break


if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('Usage: python tools/course.py DAY [options]')
    day=int(sys.argv.pop(1));run_day(day)

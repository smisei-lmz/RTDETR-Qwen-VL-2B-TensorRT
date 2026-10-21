"""Record CPU/rule/source checks. GPU, DDS and hardware need separate validation."""
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from check_sources import check


def main():
    directory = ROOT/'validation/detr-workstation'
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (directory/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    def run(name, args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True,
            encoding='utf-8', errors='replace', timeout=60,
            env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        (directory/name).write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {directory/name}')
        return result
    static = check()
    save('source-check.json', static)
    counts = {}
    for name, folder in [('project','tests'), ('ros_independent','ros_ws/tests')]:
        result = run(name+'.log', [sys.executable,'-m','unittest','discover','-s',folder,'-v'])
        count = int(re.search(r'Ran (\d+) tests',result.stderr).group(1))
        if 'skipped=' in result.stderr:
            raise RuntimeError('CPU binary missing or test skipped; not a full local pass')
        counts[name] = count
    ext = '.exe' if os.name == 'nt' else ''
    run('cpp-core.log',[ROOT/f'build/cpu/core_tests{ext}'])
    result = run('cpu-overload.log',[ROOT/f'build/cpu/edge_pipeline{ext}',
        '--frames','120','--fps','240','--work-ms','60','--lanes','1','--queue','2','--snapshot-hz','0'])
    overload = json.loads(result.stderr.splitlines()[-1])
    assert overload['all_slots_free']
    assert overload['generated'] == overload['completed']+overload['drop_oldest']+overload['drop_no_slot']
    assert overload['queue_highwater'] <= 2
    run('rule-demo.log',[sys.executable,'tools/workstation_demo.py'])
    report = dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
        change='RT-DETRv2-S and workstation departure inspection', platform=platform.platform(),
        python=platform.python_version(),
        compiler='MSVC 2022 C++17 /O1, validation/build_cpu.cmd; direct build, not CMake',
        tests=dict(counts, cpp_core='PASS', skipped=0),
        source_checks={k:v for k,v in static.items() if k!='handbook_excerpts'},
        handbook_excerpts=len(static['handbook_excerpts']), cpu_overload=overload,
        rule_demo='MOCK_RULE_ONLY; state logic only, no detector/VLM',
        not_run=['WSL/CMake build','RT-DETR PyTorch model inference',
                 'ONNX export and ONNX Runtime parity','CUDA compilation/kernels',
                 'TensorRT build/inference and COCO mAP','ROS2 colcon and live DDS/Action',
                 'Real Qwen2-VL','Real workstation video end-to-end',
                 'Jetson camera/DMABUF/EGL','30-minute GPU soak and thermal/power profiling'])
    save('results.json',report)
    (ROOT/'VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

"""Portable source/course checks; does not claim GPU or ROS runtime validation."""
import ast
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'build', 'data', 'runs', 'artifacts', '.venv', '__pycache__', 'install', 'log'}


def check():
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
             and not p.name.startswith('.')]
    python_files = [p for p in files if p.suffix == '.py']
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    xml_files = [p for p in files if p.name == 'package.xml']
    for path in xml_files:
        ET.parse(path)
    curriculum = json.loads((ROOT/'curriculum.json').read_text(encoding='utf-8'))
    assert [d['day'] for d in curriculum] == list(range(1, 38))
    for day in curriculum:
        folder = ROOT/'days'/f'day{day["day"]:02d}'
        for name in ('run.py', 'README.md', '学习入口.md'):
            assert (folder/name).is_file(), folder/name
        for source in day['sources']:
            assert (ROOT/source).is_file(), source
    handbook = (ROOT/'每日学习与代码手册.md').read_text(encoding='utf-8')
    pattern = r'摘自 \[([^\]]+)\]\([^\)]+\) 第(\d+)行起，完整实现可点击打开。\*\*这段是源码节选，不是另一个独立程序。\*\*\n\n```\w+\n(.*?)\n```'
    excerpts = []
    for source, number, text in re.findall(pattern, handbook, re.S):
        lines = (ROOT/source).read_text(encoding='utf-8-sig').splitlines()
        start = int(number)-1
        expected = '\n'.join(lines[start:start+len(text.splitlines())])
        assert expected.rstrip() == text.rstrip(), f'Stale excerpt: {source}:{number}'
        excerpts.append(dict(source=source, line=int(number),
                             sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
    assert len(excerpts) == 37, f'Expected 37 source excerpts, got {len(excerpts)}'
    assert len(re.findall(r'<a id="day\d+"></a>', handbook)) == 37
    return dict(python_ast_files=len(python_files), package_xml_files=len(xml_files),
                daily_entrypoints=37, handbook_excerpts=excerpts)


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))

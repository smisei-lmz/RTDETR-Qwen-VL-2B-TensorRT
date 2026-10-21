import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from assets import fixtures
from model_tools import compare_boxes


class PureTests(unittest.TestCase):
    def test_fixtures_have_valid_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=fixtures(tmp)
            self.assertEqual(len(rows),6)
            for row in rows:
                parts=(Path(tmp)/row['file']).read_bytes().split(b'\n',3)
                self.assertEqual(parts[0],b'P6');self.assertEqual(parts[2],b'255')
                self.assertEqual(len(parts[3]),row['width']*row['height']*3)

    def test_box_parity_is_not_vacuous(self):
        box={'class_id':0,'score':.9,'xyxy':[0,0,10,10]}
        self.assertTrue(compare_boxes([box],[box])['pass_'])
        with self.assertRaises(RuntimeError):compare_boxes([],[])
        with self.assertRaises(RuntimeError):compare_boxes([box],[])
        with self.assertRaises(RuntimeError):compare_boxes([box],[dict(box,class_id=1)])
        with self.assertRaises(RuntimeError):compare_boxes([box],[dict(box,xyxy=[50,50,60,60])])

    def test_all_days_and_entrypoints_exist(self):
        manifest=json.loads((ROOT/'curriculum.json').read_text(encoding='utf-8'))
        self.assertEqual([row['day'] for row in manifest],list(range(1,38)))
        for row in manifest:
            folder=ROOT/'days'/f'day{row["day"]:02d}'
            self.assertTrue((folder/'README.md').is_file())
            self.assertIn(f'run_day({row["day"]})',(folder/'run.py').read_text(encoding='utf-8'))
            for source in row['sources']:self.assertTrue((ROOT/source).is_file(),source)


binary=Path(os.environ.get('EDGE_TEST_BINARY',str(ROOT/'build/cpu'/('edge_pipeline.exe' if os.name=='nt' else 'edge_pipeline'))))


@unittest.skipUnless(binary.is_file(),'Build the C++ CPU target, or set EDGE_TEST_BINARY')
class CppTests(unittest.TestCase):
    def call(self,*args,expected=0):
        result=subprocess.run([str(binary),*map(str,args)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,expected,result.stderr)
        records=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        summary=json.loads(result.stderr.splitlines()[-1]) if expected==0 else None
        return records,summary,result

    def test_serial_and_snapshot_identity(self):
        records,summary,_=self.call('--serial','--frames',8,'--snapshot-hz',1000)
        self.assertEqual(summary['completed'],8);self.assertTrue(summary['all_slots_free'])
        self.assertEqual([row['frame_id'] for row in records],list(range(8)))
        image=base64.b64decode(records[0]['snapshot_ppm'],validate=True)
        self.assertTrue(image.startswith(b'P6\n640 360\n255\n'))
        self.assertEqual(records[0]['boxes'][0]['xyxy'],[0,120,128,210])

    def test_overload_bounded_and_conserved(self):
        records,s,_=self.call('--frames',100,'--fps',240,'--work-ms',60,'--lanes',1,'--queue',2)
        self.assertGreater(s['drop_oldest'],0);self.assertLessEqual(s['queue_highwater'],2)
        self.assertEqual(s['generated'],s['completed']+s['drop_oldest']+s['drop_no_slot'])
        self.assertEqual(len({(r['session_id'],r['frame_id']) for r in records}),len(records))

    def test_ppm_frame_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixtures_data=fixtures(tmp)
            records,_,_=self.call('--source',tmp,'--serial','--frames',6)
            for record in records:self.assertEqual(record['boxes'][0]['xyxy'],fixtures_data[record['frame_id']]['expected_box'])

    def test_early_recycle_is_rejected(self):
        _,_,result=self.call('--fault','early-recycle','--frames',20,expected=2)
        self.assertIn('stale/early-recycled lease',result.stderr)

    def test_repeat_start_stop(self):
        for _ in range(3):
            _,s,_=self.call('--frames',12,'--fps',240)
            self.assertTrue(s['all_slots_free'])


if __name__=='__main__':unittest.main()

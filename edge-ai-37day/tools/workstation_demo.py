"""Standard-library state-machine demo. No camera, DETR or visual understanding."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros_ws/src/edge_assistant'))
from edge_assistant.workstation import WorkstationMonitor, Observation, EventStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT/'runs/workstation-rule-demo'))
    args = parser.parse_args()
    monitor, store = WorkstationMonitor(2, 3, 2, 2), EventStore(args.out)
    # Initially empty -> occupied -> brief missed observation -> return -> leave.
    presence = [False]*3 + [True]*4 + [False]*2 + [True]*2 + [False]*8
    rows, events = [], []
    for second, present in enumerate(presence):
        stamp = 1_000_000_000_000+second*1_000_000_000
        event = monitor.update(Observation('RULE_DEMO_NOT_A_CAMERA', second, stamp, present), stamp)
        rows.append(dict(second=second, person_present=present, state=monitor.state))
        if event:
            events.append(event)
            store.write(event, 'MOCK_RULE_ONLY', answer='', reason='Synthetic observations; no detector or VLM ran')
    result = dict(mode='MOCK_RULE_ONLY', events=len(events), observations=rows)
    Path(args.out, 'timeline.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if len(events) != 1:
        raise RuntimeError('Expected exactly one confirmed-departure event')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

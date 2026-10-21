"""Day 08: 采集线程与处理线程. See README.md for prerequisites and acceptance."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from course import run_day

if __name__ == "__main__":
    try:
        run_day(8)
    except KeyboardInterrupt:
        print("Stopped by user. Review results; stopping is not an acceptance pass.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Day 08 failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

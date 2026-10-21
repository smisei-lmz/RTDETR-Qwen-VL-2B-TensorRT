"""Day 31: 板卡软件与相机能力盘点. See README.md for prerequisites and acceptance."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from course import run_day

if __name__ == "__main__":
    try:
        run_day(31)
    except KeyboardInterrupt:
        print("Stopped by user. Review results; stopping is not an acceptance pass.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Day 31 failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Portable teaching model. No CUDA, V4L2, DMA-BUF or neural network is used.

Modes: serial, threaded, async. A Future models device completion in async mode.
Run with --inject-early-recycle to deliberately demonstrate stale buffer access.
"""
import argparse
import csv
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Lease:
    slot: int
    generation: int
    frame: int
    captured_ns: int


class Pool:
    def __init__(self, count):
        self.lock = threading.Lock()
        self.slots = [dict(state='FREE', generation=0, frame=-1) for _ in range(count)]
        self.trace = []

    def event(self, lease, state):
        self.trace.append(dict(time_ns=time.monotonic_ns(), frame_id=lease.frame,
                               slot=lease.slot, generation=lease.generation, state=state))

    def acquire(self, frame):
        with self.lock:
            for i, slot in enumerate(self.slots):
                if slot['state'] == 'FREE':
                    slot.update(state='READY', generation=slot['generation'] + 1, frame=frame)
                    lease = Lease(i, slot['generation'], frame, time.monotonic_ns())
                    self.event(lease, 'READY')
                    return lease
        return None

    def check_locked(self, lease, expected):
        slot = self.slots[lease.slot]
        if (slot['generation'], slot['frame'], slot['state']) != (
                lease.generation, lease.frame, expected):
            raise RuntimeError(f'stale/invalid lease {lease}; current={slot}; expected={expected}')

    def move(self, lease, before, after):
        with self.lock:
            self.check_locked(lease, before)
            self.slots[lease.slot]['state'] = after
            self.event(lease, after)

    def verify_read(self, lease):
        with self.lock:
            self.check_locked(lease, 'READING')

    def all_free(self):
        with self.lock:
            return all(s['state'] == 'FREE' for s in self.slots)


def run(args):
    pool = Pool(args.pool)
    ready = queue.Queue(maxsize=args.queue)
    # Return records cannot be dropped: one record per live pool slot at most.
    returned = queue.Queue(maxsize=args.pool)
    capture_done = threading.Event()
    stop = threading.Event()
    stats = dict(generated=0, completed=0, drop_oldest=0, drop_no_slot=0,
                 max_ready=0, max_inflight=0, errors=[])
    latencies = []
    stats_lock = threading.Lock()

    def bump(key, value=1):
        with stats_lock:
            stats[key] += value

    def capture_loop():
        next_capture = time.monotonic()
        try:
            for frame in range(args.frames):
                if stop.is_set():
                    break
                remaining = next_capture - time.monotonic()
                if remaining > 0:
                    stop.wait(remaining)
                next_capture += 1 / args.fps
                while True:
                    try:
                        lease = returned.get_nowait()
                    except queue.Empty:
                        break
                    pool.move(lease, 'RETURN_PENDING', 'FREE')
                bump('generated')
                lease = pool.acquire(frame)
                if lease is None:
                    bump('drop_no_slot')
                    continue
                # Single producer. Removing a queued item gives this producer
                # space even if the consumer concurrently takes another item.
                try:
                    ready.put_nowait(lease)
                except queue.Full:
                    try:
                        old = ready.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        pool.move(old, 'READY', 'FREE')
                        bump('drop_oldest')
                    ready.put_nowait(lease)
                with stats_lock:
                    stats['max_ready'] = max(stats['max_ready'], ready.qsize())
        except Exception as exc:
            with stats_lock:
                stats['errors'].append(str(exc))
            stop.set()
        finally:
            capture_done.set()

    def read_work(lease):
        pool.verify_read(lease)
        # Simulate a device reading the same storage across a nonzero interval.
        time.sleep(args.work_ms / 1000)
        pool.verify_read(lease)
        return lease.frame * 2

    def consume_completed(lease, value):
        if value != lease.frame * 2:
            raise RuntimeError('frame/result mismatch')
        latencies.append((time.monotonic_ns() - lease.captured_ns) / 1e6)
        pool.move(lease, 'READING', 'RETURN_PENDING')
        returned.put_nowait(lease)
        bump('completed')

    start = time.monotonic()
    if args.mode == 'serial':
        for frame in range(args.frames):
            lease = pool.acquire(frame)
            bump('generated')
            pool.move(lease, 'READY', 'READING')
            if args.inject_early_recycle:
                pool.move(lease, 'READING', 'FREE')
            consume_completed(lease, read_work(lease))
            pool.move(returned.get_nowait(), 'RETURN_PENDING', 'FREE')
    else:
        capture = threading.Thread(target=capture_loop, name='Capture')
        capture.start()
        inflight = []
        workers = args.inflight if args.mode == 'async' else 1
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='DeviceModel') as executor:
                while not (capture_done.is_set() and ready.empty() and not inflight):
                    if stop.is_set():
                        break
                    for item in list(inflight):
                        lease, future = item
                        if future.done():
                            consume_completed(lease, future.result())
                            inflight.remove(item)
                    if len(inflight) < workers:
                        try:
                            lease = ready.get(timeout=0.001)
                        except queue.Empty:
                            continue
                        pool.move(lease, 'READY', 'READING')
                        if args.inject_early_recycle:
                            # Intentionally illegal: freeing storage before the
                            # simulated reader is done must fail deterministically.
                            pool.move(lease, 'READING', 'FREE')
                        inflight.append((lease, executor.submit(read_work, lease)))
                        stats['max_inflight'] = max(stats['max_inflight'], len(inflight))
                    else:
                        time.sleep(0.0005)
        finally:
            stop.set()
            capture.join()
        # Capture owner has exited; shutdown thread now owns pool teardown.
        while not returned.empty():
            pool.move(returned.get_nowait(), 'RETURN_PENDING', 'FREE')
    if stats['errors']:
        raise RuntimeError('; '.join(stats['errors']))
    assert pool.all_free(), 'leaked buffer lease'
    assert stats['generated'] == stats['completed'] + stats['drop_oldest'] + stats['drop_no_slot']
    assert stats['max_ready'] <= args.queue
    latencies.sort()
    def percentile(p):
        return round(latencies[min(len(latencies)-1, int((len(latencies)-1)*p))], 3) if latencies else None
    stats.update(mode=args.mode, elapsed_s=round(time.monotonic()-start, 3),
                 latency_ms_p50=percentile(.50), latency_ms_p99=percentile(.99),
                 all_slots_free=True, simulated_devices=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'trace.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time_ns','frame_id','slot','generation','state'])
        writer.writeheader()
        writer.writerows(pool.trace)
    (out/'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['serial','threaded','async'], default='async')
    p.add_argument('--frames', type=int, default=180)
    p.add_argument('--fps', type=float, default=120)
    p.add_argument('--work-ms', type=float, default=20)
    p.add_argument('--pool', type=int, default=8)
    p.add_argument('--queue', type=int, default=2)
    p.add_argument('--inflight', type=int, default=2)
    p.add_argument('--output', default='runs/lifecycle')
    p.add_argument('--inject-early-recycle', action='store_true')
    args = p.parse_args()
    if min(args.frames, args.fps, args.pool, args.queue, args.inflight) <= 0 or args.work_ms < 0:
        p.error('counts and fps must be positive; work-ms must be nonnegative')
    try:
        run(args)
    except (RuntimeError, AssertionError) as exc:
        print(json.dumps(dict(error=str(exc), expected_fault=args.inject_early_recycle)))
        raise SystemExit(2)


if __name__ == '__main__':
    main()

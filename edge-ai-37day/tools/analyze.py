import argparse
import json
import statistics
from pathlib import Path


def analyze(path,warmup=10):
    records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    results=[row for row in records if row.get('type')=='result']
    measured=results[warmup:]
    if not measured:raise RuntimeError('Not enough completed frames after warmup')
    values=sorted(row['latency_ms'] for row in measured)
    percentile=lambda fraction:values[int((len(values)-1)*fraction)]
    ages=[row['captured_mono_ns'] for row in measured]
    interval=(max(ages)-min(ages))/1e9
    return dict(completed_samples=len(measured),warmup_removed=warmup,
        latency_mean_ms=statistics.mean(values),p50_ms=percentile(.5),p95_ms=percentile(.95),p99_ms=percentile(.99),
        completion_density_per_source_second=(len(measured)-1)/interval if interval>0 else None,
        device_span_mean_ms=statistics.mean(row['device_ms'] for row in measured),
        stage_mean_ms={key:statistics.mean(row.get(key,0) for row in measured) for key in
            ['copy_or_yuv_ms','preprocess_ms','inference_ms','gpu_post_d2h_ms','cpu_ms']},
        note='device_ms is the event span including transfers/kernels, not isolated TensorRT latency; '
             'density uses source timestamps and is not a steady-state throughput measurement by itself')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('events');p.add_argument('--warmup',type=int,default=10);p.add_argument('--out')
    a=p.parse_args();report=analyze(a.events,a.warmup);text=json.dumps(report,indent=2)
    if a.out:Path(a.out).write_text(text,encoding='utf-8')
    print(text)

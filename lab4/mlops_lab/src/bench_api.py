import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import TEXT_COL, split

URL = 'http://127.0.0.1:8000/predict'
REPEATS = 50
texts = split()[2][TEXT_COL].tolist()
rows = []
for batch in (1, 10, 50, 100):
    body = json.dumps({'texts': texts[:batch]}).encode()
    handler, total = [], []
    for _ in range(REPEATS):
        req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)
        total.append((time.perf_counter() - t0) * 1000)
        handler.append(res['latency_ms'])
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    row = {'batch': batch, 'handler_median_ms': round(statistics.median(handler), 3),
           'handler_p95_ms': round(q(handler, .95), 3), 'client_median_ms': round(statistics.median(total), 3),
           'client_p95_ms': round(q(total, .95), 3), 'per_message_ms': round(statistics.median(total) / batch, 3)}
    rows.append(row)
    print(row)
Path('report/analysis/bench.json').write_text(json.dumps(rows, indent=2))

import json
import urllib.request
from pathlib import Path

body = Path('sample_request.json').read_bytes()
req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body,
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=30) as r:
    result = json.load(r)
expected = json.loads(Path('expected_predictions.json').read_text())
assert result['predictions'] == expected, (result, expected)
print('PASS: container predictions match exported model', result)

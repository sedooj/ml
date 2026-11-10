"""Автоматическая проверка контракта API: некорректные запросы -> 422 с ожидаемым типом ошибки,
корректный запрос из одного SMS -> 200."""
import json
import sys
import urllib.error
import urllib.request

URL = 'http://127.0.0.1:8000/predict'
CASES = [
    ('пустой пакет', b'{"texts": []}', 422, 'too_short'),
    ('нет поля texts', b'{"text": ["hello"]}', 422, 'missing'),
    ('строка из пробелов', b'{"texts": ["   "]}', 422, 'string_too_short'),
    ('лишнее поле', b'{"texts": ["Call now!"], "debug": true}', 422, 'extra_forbidden'),
    ('число вместо строки', b'{"texts": [123]}', 422, 'string_type'),
    ('строка вместо списка', b'{"texts": "WINNER!"}', 422, 'list_type'),
    ('101 сообщение', json.dumps({'texts': ['hi'] * 101}).encode(), 422, 'too_long'),
    ('строка длиннее 1000', json.dumps({'texts': ['a' * 1001]}).encode(), 422, 'string_too_long'),
    ('некорректный JSON', b'not a json', 422, 'json_invalid'),
    ('корректный запрос', b'{"texts": ["Hey, are we still meeting for lunch tomorrow?"]}', 200, None),
]


def call(body):
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


failed = 0
for name, body, code, err in CASES:
    status, res = call(body)
    types = [d['type'] for d in res['detail']] if status == 422 else []
    ok = status == code and (err is None or err in types)
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {status}  {name:<22} {', '.join(types) or res.get('labels')}")
print(f'{len(CASES) - failed}/{len(CASES)} checks passed')
sys.exit(1 if failed else 0)

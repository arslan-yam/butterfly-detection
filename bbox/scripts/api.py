import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://tasks.yandex.ru/api/v1"


def get_key():
    key = os.environ.get("YT_API_KEY")
    if not key:
        sys.exit("no YT_API_KEY, run: export YT_API_KEY=<key>")
    return key


def call(method, path, key, scheme, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"{scheme} {key}", "Content-Type": "application/json"}
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except OSError as e:
        return None, f"network error: {e}"


def find_scheme(key, test_path):
    for scheme in ("OAuth", "ApiKey"):
        code, resp = call("GET", test_path, key, scheme)
        if code == 200:
            return scheme, resp
    sys.exit(f"key does not work ({code}): {resp}")


def set_status(aid, status, key, scheme, comment=None):
    code, cur = call("GET", f"/assignments/{aid}", key, scheme)
    if code != 200 or cur.get("status") != "SUBMITTED":
        return "skipped"

    body = {"status": status}
    if comment:
        body["public_comment"] = comment[:2048]
    code, resp = call("PATCH", f"/assignments/{aid}", key, scheme, body)

    if code is None:
        code, cur = call("GET", f"/assignments/{aid}", key, scheme)
        if code == 200 and cur.get("status") != status:
            code = None
    if code == 200:
        return "ok"
    print(f"  error {aid}: {code} {resp}")
    return "error"

#!/usr/bin/env python3
"""Run all LeetGPU solutions against their challenges (action=run, never submit).

Usage: run_all.py <session.json> [challenge_id ...]
  session.json: {"access_token": "...", "refresh_token": "..."} (from browser localStorage)
"""
import asyncio
import json
import os
import sys
import time
import urllib.request
import uuid

import websockets

WS_URL = "wss://api.leetgpu.com/api/v1/ws/submit"
SUPABASE_URL = "https://yhdtysacdkqoquvkdwdd.supabase.co"
ANON_KEY = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InloZHR5c2FjZGtxb3F1dmtkd2RkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg2MzQ3MTksImV4cCI6MjA1NDIxMDcxOX0."
            "aevKbBm0HgYbEI6CQ6UobY728lYwhi7dawnI3F_d0QM")
SRC = "/home/young/code/Young-TW/LeetGPU/src"
SCRATCH = os.path.dirname(os.path.abspath(__file__))


class Session:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            d = json.load(f)
        self.access_token = d["access_token"]
        self.refresh_token = d.get("refresh_token")
        self.expires_at = d.get("expires_at", 0)
        self.user_id = d.get("user", {}).get("id")

    def refresh(self):
        req = urllib.request.Request(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            data=json.dumps({"refresh_token": self.refresh_token}).encode(),
            headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
            method="POST")
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        self.access_token = d["access_token"]
        self.refresh_token = d.get("refresh_token", self.refresh_token)
        self.expires_at = d.get("expires_at", time.time() + d.get("expires_in", 3600))
        if d.get("user"):
            self.user_id = d["user"].get("id")
        with open(self.path, "w") as f:
            json.dump({"access_token": self.access_token, "refresh_token": self.refresh_token,
                       "expires_at": self.expires_at, "user": {"id": self.user_id}}, f)

    def ensure_fresh(self):
        if self.refresh_token and (not self.expires_at or time.time() > self.expires_at - 120):
            self.refresh()


async def run_one(sess: Session, challenge_id: int, path: str, language: str,
                  accelerator: str = "T4", timeout: float = 240.0):
    name = "solution.cu" if language == "cuda" else "solution.py"
    with open(path) as f:
        content = f.read()

    msg = {
        "token": sess.access_token,
        "submissionId": str(uuid.uuid4()),
        "action": "run",
        "submission": {
            "files": [{"name": name, "content": content}],
            "language": language,
            "accelerator": accelerator,
            "mode": "accelerated",
            "challengeId": challenge_id,
            "userId": sess.user_id,
            "public": False,
        },
    }

    output = []
    final = "no-final-status"
    async with websockets.connect(WS_URL, open_timeout=30) as ws:
        await ws.send(json.dumps(msg))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                ev = json.loads(raw)
                if ev.get("output"):
                    output.append(ev["output"])
                status = ev.get("status")
                if status and status != "running":
                    final = status
                    break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
            final = f"transport: {e!r}"
    return final, "".join(output)


async def main():
    sess = Session(sys.argv[1])
    index = json.load(open(os.path.join(SCRATCH, "index.json")))  # [id, title, diff, slug]
    only = {int(a) for a in sys.argv[2:]} if len(sys.argv) > 2 else None

    results_path = os.path.join(SCRATCH, "run_results.json")
    results = {}
    if os.path.exists(results_path):
        results = json.load(open(results_path))

    todo = [(i, sl) for i, _, _, sl in index
            if (only is None or i in only) and str(i) not in results]
    for n, (cid, slug) in enumerate(todo):
        language = "pytorch" if cid == 41 else "cuda"
        ext = "py" if cid == 41 else "cu"
        path = os.path.join(SRC, f"{slug}.{ext}")
        sess.ensure_fresh()
        try:
            status, out = await run_one(sess, cid, path, language)
        except Exception as e:
            status, out = f"exception: {e!r}", ""
        results[str(cid)] = {"slug": slug, "status": status, "output": out}
        with open(results_path, "w") as f:
            json.dump(results, f)
        print(f"[{n+1}/{len(todo)}] #{cid} {slug}: {status}", flush=True)
        if status not in ("success",):
            print("\n".join(out.splitlines()[-12:]), flush=True)
        await asyncio.sleep(2.0)

    ok = sum(1 for r in results.values() if r["status"] == "success")
    print(f"\nDone: {ok}/{len(results)} success")
    for cid, r in sorted(results.items(), key=lambda kv: int(kv[0])):
        if r["status"] != "success":
            print(f"  #{cid} {r['slug']}: {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())

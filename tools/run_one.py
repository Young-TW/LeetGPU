#!/usr/bin/env python3
"""Run a single LeetGPU solution via the ws/submit websocket (action=run)."""
import asyncio
import json
import sys
import uuid

import websockets

WS_URL = "wss://api.leetgpu.com/api/v1/ws/submit"


async def run(challenge_id: int, path: str, language: str, mode: str, token: str | None,
              accelerator: str = "T4", timeout: float = 180.0):
    name = "solution.cu" if language == "cuda" else "solution.py"
    with open(path) as f:
        content = f.read()

    msg = {
        "token": token,
        "submissionId": str(uuid.uuid4()),
        "action": "run",
        "submission": {
            "files": [{"name": name, "content": content}],
            "language": language,
            "accelerator": accelerator,
            "mode": mode,
            "challengeId": challenge_id,
            "userId": None,
            "public": False,
        },
    }

    events = []
    async with websockets.connect(WS_URL, open_timeout=30) as ws:
        await ws.send(json.dumps(msg))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                ev = json.loads(raw)
                events.append(ev)
                status = ev.get("status")
                if status and status != "running":
                    break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
            events.append({"status": "transport", "output": repr(e)})
    return events


if __name__ == "__main__":
    cid = int(sys.argv[1])
    path = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    mode = sys.argv[4] if len(sys.argv) > 4 else "functional"
    token = sys.argv[5] if len(sys.argv) > 5 else None
    evs = asyncio.run(run(cid, path, language, mode, token))
    for ev in evs:
        print(json.dumps(ev)[:2000])

#!/usr/bin/env python3
"""Compile-check every CUDA solution via LeetGPU playground functional mode."""
import asyncio
import json
import os
import sys
import uuid

import websockets

WS_URL = "wss://api.leetgpu.com/api/v1/ws/submit"
SRC = "/home/young/code/Young-TW/LeetGPU/src"
MAIN_STUB = "int main() { return 0; }\n"


async def compile_one(path: str, timeout: float = 120.0) -> tuple[str, str]:
    with open(path) as f:
        content = f.read()

    msg = {
        "token": None,
        "submissionId": str(uuid.uuid4()),
        "action": "run",
        "submission": {
            "files": [
                {"name": "solution.cu", "content": content},
                {"name": "main.cu", "content": MAIN_STUB},
            ],
            "language": "cuda",
            "accelerator": "GTX TITAN X",
            "mode": "functional",
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
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".cu"))
    if len(sys.argv) > 1:
        files = [f for f in files if f in sys.argv[1:]]

    results = {}
    for i, fn in enumerate(files):
        try:
            status, out = await compile_one(os.path.join(SRC, fn))
        except Exception as e:
            status, out = f"exception: {e!r}", ""
        ok = status == "success"
        results[fn] = {"status": status, "output": out}
        print(f"[{i+1}/{len(files)}] {fn}: {status}", flush=True)
        if not ok:
            # keep only the interesting part of compiler noise
            tail = "\n".join(out.splitlines()[:20])
            print(tail, flush=True)
        await asyncio.sleep(1.0)

    with open("compile_results.json", "w") as f:
        json.dump(results, f)
    bad = [fn for fn, r in results.items() if r["status"] != "success"]
    print(f"\nDone. {len(results) - len(bad)}/{len(results)} compiled OK.")
    if bad:
        print("Failed:", *bad, sep="\n  ")


if __name__ == "__main__":
    asyncio.run(main())

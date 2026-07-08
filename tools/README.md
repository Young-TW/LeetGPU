# tools/

Scripts and data for fetching, checking, and running the LeetGPU solutions.

## Data

- `index.json` — `[id, title, difficulty, slug]` for all 89 challenges
- `details.json` — full challenge specs + starter code (all languages), fetched from
  `https://api.leetgpu.com/api/v1/challenges/{id}`

## Remote (leetgpu.com)

Requires `pip install websockets`.

- `compile_check.py` — compile-checks `src/*.cu` on LeetGPU's **functional simulator**
  (no login needed). Note: the simulator lacks some CUDA intrinsics (`rsqrtf`, `__brev`,
  `half`, …) — solutions avoid them where possible; the 3 FP16 problems can't compile there.
- `run_one.py <id> <file> [language] [mode] [token]` — run one solution over the websocket.
- `run_all.py <session.json> [ids...]` — run every solution against its challenge on a real
  GPU (`action: "run"`, never submits). Needs a login token: on leetgpu.com run
  `copy(localStorage.getItem('sb-yhdtysacdkqoquvkdwdd-auth-token'))` in the browser console
  and save the JSON to `session.json`. Auto-refreshes expired tokens; results are written to
  `run_results.json` next to the script.

## Local (ROCm)

`ROCm/*.hip` are hipify-perl conversions of `src/*.cu`; all 88 compile with
`hipcc --offload-arch=gfx1201` (RX 9070 XT).

- `smoke_main.hip` — example harness: builds the Vector Addition spec example, calls
  `solve`, checks the output.

  ```sh
  hipcc tools/smoke_main.hip ROCm/vector_addition.hip -o smoke --offload-arch=gfx1201 -w && ./smoke
  ```

Sample test data lives in the problem specs (`problems/*.md`, Examples sections):
86/89 problems have concrete input/output pairs; #20 (K-Means), #74 (GPT-2 Block) and
#93 (Llama Block) do not.

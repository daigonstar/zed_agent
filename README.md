# Zed Ollama agent

An [ACP](https://agentclientprotocol.com) agent that lets Zed drive a local Ollama model
(default `qwen3.6-coding-agent`, see below) with sandboxed, approval-gated tools.

## Model

`install.sh` pulls `qwen3.6:35b-a3b-coding` and builds `qwen3.6-coding-agent` from it:
the same weights without the vision projector and with MTP drafting turned off, plus
`num_ctx 65536` and `num_batch 1024`. On a 16 GB card (RX 7800 XT, 32 GB RAM) the stock tag
overfills VRAM and its MTP draft step stalls generation at about 1 token/s; the tuned
variant measured about 1,250 tokens/s reading (a 31K-token prompt in 25 s) and about
50 tokens/s writing, with Ollama keeping 22 of 41 layers' experts in VRAM and the rest in
system RAM.

Ollama service settings (`/etc/systemd/system/ollama.service.d/override.conf`):

    [Service]
    Environment="OLLAMA_FLASH_ATTENTION=1"
    Environment="LLAMA_ARG_FIT_TARGET=3072"
    Environment="OLLAMA_NUM_PARALLEL=1"
    Environment="LLAMA_ARG_CACHE_RAM=1024"
    Environment="LLAMA_ARG_CTX_CHECKPOINTS=8"

`LLAMA_ARG_CACHE_RAM=1024` and `LLAMA_ARG_CTX_CHECKPOINTS=8` keep long sessions from
failing. By default llama-server saves every prompt it moves away from (about 600 MB at
27K tokens) in a RAM cache of up to 8 GB, plus up to 32 context checkpoints of about 60 MB
each. Each history trim saves one. In testing the cache reached 7.7 GB and pushed the
model's CPU-side experts out of memory. Generation then stalled for minutes, one cache save
took 500 s, and Ollama returned a 500 error after 5 minutes. With a `MemoryMax` on the
service this happens sooner. Old prompts are almost never reused here, so 1 GB is enough.

`LLAMA_ARG_FIT_TARGET=3072` matters: Ollama's automatic placement underestimates this
model's VRAM use by about 1.5 GB, so with the default margin the card ends up nearly full
and the driver starts evicting GPU memory to system RAM. Generation then slows to a crawl
mid-session, and in the worst case RAM runs out and the desktop crashes. With a 3 GB margin
about 2 GB stays free and speed is unchanged.

- Do not set `num_gpu` (in a Modelfile or request). It turns off the automatic placement;
  pinning 30 layers overfilled VRAM and crashed Hyprland in testing.
- Do not set `OLLAMA_MAX_LOADED_MODELS=1`: the screenshot model would unload the main model.

## Install (also after a reinstall)

    ./install.sh              # venv + deps + bubblewrap + model + Zed settings + self-check
    ./install.sh --dry-run    # show what would happen
    ./install.sh --no-model --no-system-deps --no-zed-config   # only the Python side

Options: `--model NAME`, `--skip-tests`. Everything is idempotent. Zed's `settings.json` is
backed up to `settings.json.bak-zed-agent` before it is changed; env overrides you made
under the "Hypr Agent" entry are kept.

## Files

| Path | Purpose |
| --- | --- |
| `hypr_agent_zed.py` | The agent (run `--check` for settings and sandbox probes) |
| `agent_context_coding.md` | Coding guidance appended to the system prompt |
| `tests/` | Self-tests: `~/.venvs/zed-agent/bin/python -m unittest discover -s tests -v` |
| `evals/run_evals.py` | End-to-end tasks against the real model over ACP (Laravel, dotfiles, navigation) |
| `requirements.txt` | Pinned Python dependencies |
| `legacy/` | Old variant, kept for reference |

Settings are environment variables documented at the top of `hypr_agent_zed.py`
(`OLLAMA_AGENT_THINK`, `OLLAMA_AGENT_SANDBOX`, `OLLAMA_AGENT_WRITE_HOME`, ...). Set them in
the "Hypr Agent" `env` block in Zed's settings.

## Vision / screenshots

The agent advertises image support to Zed, so you can paste or attach screenshots to a message. Images are used for that turn only (not kept in history). When the main model has vision, images go to it directly. Otherwise (the default `qwen3.6-coding-agent` is text-only) they are described by `OLLAMA_AGENT_VISION_MODEL` (default `qwen2.5vl:7b`) and the description is passed on as text. The vision model runs on the CPU so the main model stays loaded on the GPU with its prompt cache; this takes about 30-80 s per image. Set `OLLAMA_AGENT_VISION_GPU=1` to run it on the GPU instead (faster, but it unloads the main model), or `OLLAMA_AGENT_VISION_MODEL=off` to disable. Limits: 4 images per message, 10 MB each.

## Running-app checks (Laravel / Vite)

- `dev_server` starts `npm run dev`, `npx vite` or `php artisan serve` in the sandbox (localhost only, approval required) and keeps it running; `logs`, `status` and `stop` manage it. Servers are stopped when the agent exits.
- `check_app` opens a `http://localhost:PORT/...` page: HTTP status and Laravel/Vite error pages, plus a screenshot (attached for a vision model, otherwise described as text). It uses Chromium first (console errors, rendered DOM, screenshot), then Zen (`zen-browser`) or Firefox (screenshot only). Set `OLLAMA_AGENT_BROWSER` to override. Without a browser only the HTTP check runs. Ports not started by the agent need approval.

## Evals

`evals/run_evals.py` runs real tasks end to end: it starts the agent over ACP the way Zed
does, gives it a throwaway workspace, approves every request, and checks the files on disk.

    ~/.venvs/zed-agent/bin/python evals/run_evals.py                  # all tasks
    ~/.venvs/zed-agent/bin/python evals/run_evals.py laravel_dashboard
    OLLAMA_AGENT_MODEL=qwen3:14b ~/.venvs/zed-agent/bin/python evals/run_evals.py

Results go to `evals/results/` (ignored by git). Failed workspaces are kept for inspection.

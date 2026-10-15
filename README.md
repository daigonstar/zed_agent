# Zed Ollama agent

An [ACP](https://agentclientprotocol.com) agent that lets Zed drive a local Ollama model
(default `qwen3.5:9b-q8_0`) with sandboxed, approval-gated tools.

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
| `requirements.txt` | Pinned Python dependencies |
| `legacy/` | Old variant, kept for reference |

Settings are environment variables documented at the top of `hypr_agent_zed.py`
(`OLLAMA_AGENT_THINK`, `OLLAMA_AGENT_SANDBOX`, `OLLAMA_AGENT_WRITE_HOME`, ...). Set them in
the "Hypr Agent" `env` block in Zed's settings.

## Vision / screenshots

The agent advertises image support to Zed, so you can paste or attach screenshots to a message. Images are sent to Ollama for that turn only (not kept in history). This needs a vision-capable model, e.g. `OLLAMA_AGENT_MODEL=qwen2.5vl:7b` (also gemma3, llama3.2-vision). If the model reports no vision capability the agent says so instead of sending the image. Limits: 4 images per message, 10 MB each.

## Running-app checks (Laravel / Vite)

- `dev_server` starts `npm run dev`, `npx vite` or `php artisan serve` in the sandbox (localhost only, approval required) and keeps it running; `logs`, `status` and `stop` manage it. Servers are stopped when the agent exits.
- `check_app` opens a `http://localhost:PORT/...` page: HTTP status and Laravel/Vite error pages, plus a screenshot passed to a vision model. It uses Chromium first (console errors, rendered DOM, screenshot), then Zen (`zen-browser`) or Firefox (screenshot only). Set `OLLAMA_AGENT_BROWSER` to override. Without a browser only the HTTP check runs. Ports not started by the agent need approval.

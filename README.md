# Zed Ollama agent

An [ACP](https://agentclientprotocol.com) agent that lets Zed drive a local Ollama model
(default `qwen3:14b`) with sandboxed, approval-gated tools.

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

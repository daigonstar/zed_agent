#!/usr/bin/env bash
#
# Installs the Zed Ollama agent: Python venv, dependencies, bubblewrap sandbox,
# the model, and the "Hypr Agent" entry in Zed's settings.json. Safe to re-run.
#
# Usage: ./install.sh [--dry-run] [--no-system-deps] [--no-model] [--no-zed-config]
#                     [--model NAME] [--skip-tests]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ZED_AGENT_VENV:-$HOME/.venvs/zed-agent}"
MODEL="${OLLAMA_AGENT_MODEL:-qwen3.6-coding-agent}"
VISION_MODEL="${OLLAMA_AGENT_VISION_MODEL:-qwen2.5vl:7b}"
# qwen3.6-coding-agent is built locally from this tag (see "Model" below).
AGENT_BASE_MODEL="qwen3.6:35b-a3b-coding"
ZED_SETTINGS="${ZED_SETTINGS:-$HOME/.config/zed/settings.json}"
AGENT_NAME="Hypr Agent"

DRY_RUN=false SYSTEM_DEPS=true PULL_MODEL=true ZED_CONFIG=true RUN_TESTS=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --no-system-deps) SYSTEM_DEPS=false ;;
    --no-model) PULL_MODEL=false ;;
    --no-zed-config) ZED_CONFIG=false ;;
    --skip-tests) RUN_TESTS=false ;;
    --model) MODEL="${2:?--model needs a name}"; shift ;;
    -h|--help) sed -n 2,8p "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
run() {
  if $DRY_RUN; then printf '[dry-run] %s\n' "$*"; else "$@"; fi
}

[[ -f "$SCRIPT_DIR/hypr_agent_zed.py" ]] || { echo "hypr_agent_zed.py not found next to install.sh" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

# --- System packages ---------------------------------------------------------
step "System packages"
missing=()
command -v bwrap >/dev/null || missing+=(bubblewrap)
command -v ollama >/dev/null || missing+=(ollama)
command -v git >/dev/null || missing+=(git)
if ((${#missing[@]})); then
  if $SYSTEM_DEPS && command -v pacman >/dev/null; then
    echo "Installing: ${missing[*]}"
    run sudo pacman -S --needed --noconfirm "${missing[@]}" || warn "Could not install ${missing[*]}"
  else
    warn "Missing: ${missing[*]} (install them yourself; bubblewrap is required for the sandbox)"
  fi
else
  echo "bubblewrap, ollama and git are present."
fi

# --- Python environment ------------------------------------------------------
step "Python environment ($VENV)"
if [[ ! -x "$VENV/bin/python" ]]; then
  run mkdir -p "$(dirname "$VENV")"
  run python3 -m venv "$VENV"
fi
run "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -r "$SCRIPT_DIR/requirements.txt"

# --- Model -------------------------------------------------------------------
installed() { ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxE "$1(:latest)?"; }

pull() {
  if installed "$1"; then
    echo "$1 is already installed."
  else
    run ollama pull "$1" || { warn "Could not pull $1; run 'ollama pull $1' later."; return 1; }
  fi
}

# Build qwen3.6-coding-agent from the stock coding tag: drop the vision projector (it costs
# VRAM that would otherwise hold experts) and turn MTP drafting off (with experts offloaded
# to system RAM it stalls generation to ~1 token/s on a 16 GB card). Images are described
# by $VISION_MODEL instead.
build_agent_model() {
  local modelfile
  pull "$AGENT_BASE_MODEL" || return 1
  if $DRY_RUN; then
    echo "[dry-run] ollama create qwen3.6-coding-agent (from $AGENT_BASE_MODEL, text only, no MTP)"
    return 0
  fi
  modelfile="$(mktemp)"
  ollama show "$AGENT_BASE_MODEL" --modelfile | awk '
    /^FROM / { if (seen++) next }                # keep the weights, drop the vision projector
    /^PARAMETER (draft_num_predict|num_ctx|num_batch) / { next }
    { print }
    END {
      print "PARAMETER draft_num_predict 0"
      print "PARAMETER num_ctx 65536"
      print "PARAMETER num_batch 1024"
    }' > "$modelfile"
  ollama create qwen3.6-coding-agent -f "$modelfile" || warn "Could not create qwen3.6-coding-agent."
  rm -f "$modelfile"
}

if $PULL_MODEL; then
  step "Model ($MODEL)"
  if ! command -v ollama >/dev/null; then
    warn "ollama is not installed; skipping the model download."
  elif ! curl -fsS --max-time 3 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1 && ! $DRY_RUN; then
    warn "Ollama is not running (try: sudo systemctl enable --now ollama). Then re-run this script."
  else
    if [[ "$MODEL" == qwen3.6-coding-agent ]]; then
      if installed "$MODEL"; then echo "$MODEL is already installed."; else build_agent_model; fi
      # Without a bigger VRAM margin Ollama nearly fills a 16 GB card with this model and the
      # driver starts evicting GPU memory to RAM (see README "Model").
      if command -v systemctl >/dev/null && ! systemctl show ollama -p Environment 2>/dev/null | grep -q LLAMA_ARG_FIT_TARGET; then
        warn "The ollama service has no LLAMA_ARG_FIT_TARGET. Add Environment=\"LLAMA_ARG_FIT_TARGET=3072\" to /etc/systemd/system/ollama.service.d/override.conf, then: sudo systemctl daemon-reload && sudo systemctl restart ollama"
      fi
    else
      pull "$MODEL" || true
    fi
    if [[ -n "$VISION_MODEL" && "$VISION_MODEL" != off ]]; then
      step "Vision model for screenshots ($VISION_MODEL)"
      pull "$VISION_MODEL" || true
    fi
  fi
fi

# --- Zed settings ------------------------------------------------------------
if $ZED_CONFIG; then
  step "Zed settings ($ZED_SETTINGS)"
  ZED_SETTINGS="$ZED_SETTINGS" AGENT_NAME="$AGENT_NAME" MODEL="$MODEL" VENV="$VENV" \
  SCRIPT="$SCRIPT_DIR/hypr_agent_zed.py" DRY_RUN="$DRY_RUN" python3 - <<'PY'
import json, os, shutil, sys
from pathlib import Path

path = Path(os.environ["ZED_SETTINGS"])
entry = {
    "type": "custom",
    "command": f"{os.environ['VENV']}/bin/python",
    "args": [os.environ["SCRIPT"]],
    "env": {
        "OLLAMA_AGENT_SANDBOX": "require",
        "OLLAMA_AGENT_MODEL": os.environ["MODEL"],
    },
}
snippet = json.dumps({"agent_servers": {os.environ["AGENT_NAME"]: entry}}, indent=2)
dry = os.environ["DRY_RUN"] == "true"

try:
    data = json.loads(path.read_text()) if path.exists() else {}
except ValueError:
    print("settings.json has comments or is not plain JSON, so it was left alone.")
    print("Add this under the top level of your Zed settings:\n" + snippet)
    sys.exit(0)

servers = data.setdefault("agent_servers", {})
existing = servers.get(os.environ["AGENT_NAME"])
if existing:
    # Keep the user's own env tweaks; only refresh the command and script path.
    kept = {k: v for k, v in existing.get("env", {}).items()
            if k not in {"PYTHONPATH", "OLLAMA_AGENT_ALLOW_WRITES"}}  # no longer used
    entry["env"] = {**entry["env"], **kept}
if existing == entry:
    print("Already configured.")
    sys.exit(0)
servers[os.environ["AGENT_NAME"]] = entry
if dry:
    print("[dry-run] would write:\n" + snippet)
    sys.exit(0)
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    shutil.copy2(path, path.with_suffix(".json.bak-zed-agent"))
path.write_text(json.dumps(data, indent=2) + "\n")
print(f"Added '{os.environ['AGENT_NAME']}' (backup: {path.name}.bak-zed-agent).")
PY
fi

# --- Verify ------------------------------------------------------------------
if ! $DRY_RUN; then
  step "Self-check (effective settings and sandbox probes)"
  OLLAMA_AGENT_MODEL="$MODEL" "$VENV/bin/python" "$SCRIPT_DIR/hypr_agent_zed.py" --check || warn "Self-check reported problems."
  if $RUN_TESTS; then
    step "Self-tests"
    (cd "$SCRIPT_DIR" && "$VENV/bin/python" -m unittest discover -s tests) || warn "Some self-tests failed."
  fi
fi

step "Done. Restart Zed and pick '$AGENT_NAME' in the agent panel."

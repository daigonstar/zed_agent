#!/usr/bin/env python3
"""End-to-end evals: drive hypr_agent_zed.py over ACP (as Zed does) against a real Ollama model.

Each task builds a throwaway workspace, sends one prompt, auto-approves every permission
request, then checks the result on disk (and in the reply). Nothing outside the temporary
workspace is touched: the agent gets its own state dir and session log.

Run with the agent's interpreter:
    ~/.venvs/zed-agent/bin/python evals/run_evals.py                 # all tasks
    ~/.venvs/zed-agent/bin/python evals/run_evals.py laravel_route   # some tasks
    OLLAMA_AGENT_MODEL=qwen3:14b ~/.venvs/zed-agent/bin/python evals/run_evals.py

Results are printed as a table and written to evals/results/<model>-<time>.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from acp import spawn_agent_process, text_block
from acp.schema import AllowedOutcome, ClientCapabilities, RequestPermissionResponse

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "hypr_agent_zed.py"
TASK_TIMEOUT = int(os.environ.get("EVAL_TASK_TIMEOUT", "900"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def write_tree(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def php_lint(ws: Path, *rels: str) -> str | None:
    for rel in rels:
        result = subprocess.run(["php", "-l", str(ws / rel)], capture_output=True, text=True)
        if result.returncode != 0:
            return f"php -l {rel}: {result.stdout.strip()[-200:]}"
    return None


def laravel_skeleton(ws: Path, padding: int = 60) -> None:
    """A Laravel-shaped app (no vendor dir) with enough files to need real navigation."""
    files = {
        "composer.json": json.dumps({"name": "acme/shop", "require": {"laravel/framework": "^12.0"}}, indent=2),
        "artisan": "#!/usr/bin/env php\n<?php\n// artisan stub\n",
        "bootstrap/app.php": (
            "<?php\n\nuse Illuminate\\Foundation\\Application;\nuse Illuminate\\Foundation\\Configuration\\Middleware;\n\n"
            "return Application::configure(basePath: dirname(__DIR__))\n"
            "    ->withRouting(web: __DIR__.'/../routes/web.php', health: '/up')\n"
            "    ->withMiddleware(function (Middleware $middleware) {\n        //\n    })\n"
            "    ->create();\n"
        ),
        "routes/web.php": (
            "<?php\n\nuse App\\Http\\Controllers\\HomeController;\nuse App\\Http\\Controllers\\OrderController;\n"
            "use Illuminate\\Support\\Facades\\Route;\n\n"
            "Route::get('/', [HomeController::class, 'index'])->name('home');\n\n"
            "Route::middleware('auth')->group(function () {\n"
            "    Route::get('/orders', [OrderController::class, 'index'])->name('orders.index');\n"
            "    Route::get('/orders/{order}', [OrderController::class, 'show'])->name('orders.show');\n"
            "});\n"
        ),
        "app/Http/Controllers/Controller.php": "<?php\n\nnamespace App\\Http\\Controllers;\n\nabstract class Controller\n{\n    //\n}\n",
        "app/Http/Controllers/HomeController.php": (
            "<?php\n\nnamespace App\\Http\\Controllers;\n\nclass HomeController extends Controller\n{\n"
            "    public function index()\n    {\n        return view('home');\n    }\n}\n"
        ),
        "app/Http/Controllers/OrderController.php": (
            "<?php\n\nnamespace App\\Http\\Controllers;\n\nuse App\\Models\\Order;\nuse Illuminate\\Http\\Request;\n\n"
            "class OrderController extends Controller\n{\n"
            "    public function index(Request $request)\n    {\n"
            "        $orders = Order::where('user_id', $request->user()->id)->latest()->paginate(20);\n\n"
            "        return view('orders.index', compact('orders'));\n    }\n\n"
            "    public function show(Order $order)\n    {\n        return view('orders.show', compact('order'));\n    }\n}\n"
        ),
        "app/Models/User.php": (
            "<?php\n\nnamespace App\\Models;\n\nuse Illuminate\\Foundation\\Auth\\User as Authenticatable;\n\n"
            "class User extends Authenticatable\n{\n    protected $fillable = ['first_name', 'last_name', 'email', 'password'];\n\n"
            "    public function getFullName(): string\n    {\n        return trim($this->first_name.' '.$this->last_name);\n    }\n\n"
            "    public function orders()\n    {\n        return $this->hasMany(Order::class);\n    }\n}\n"
        ),
        "app/Models/Order.php": (
            "<?php\n\nnamespace App\\Models;\n\nuse Illuminate\\Database\\Eloquent\\Model;\n\n"
            "class Order extends Model\n{\n    protected $fillable = ['user_id', 'total', 'status'];\n\n"
            "    public function user()\n    {\n        return $this->belongsTo(User::class);\n    }\n}\n"
        ),
        "app/Services/Billing/InvoiceCalculator.php": (
            "<?php\n\nnamespace App\\Services\\Billing;\n\nclass InvoiceCalculator\n{\n"
            "    private const REGION_RATES = ['uk' => 0.20, 'de' => 0.19, 'fr' => 0.20];\n\n"
            "    public function calculateInvoiceTax(float $net, string $region): float\n    {\n"
            "        $rate = self::REGION_RATES[$region] ?? 0.0;\n\n        return round($net * $rate, 2);\n    }\n}\n"
        ),
        "app/Mail/OrderShipped.php": (
            "<?php\n\nnamespace App\\Mail;\n\nuse App\\Models\\Order;\n\nclass OrderShipped\n{\n"
            "    public function __construct(public Order $order) {}\n\n"
            "    public function subject(): string\n    {\n"
            "        return 'Order shipped for '.$this->order->user->getFullName();\n    }\n}\n"
        ),
        "app/Http/Controllers/ProfileController.php": (
            "<?php\n\nnamespace App\\Http\\Controllers;\n\nuse Illuminate\\Http\\Request;\n\n"
            "class ProfileController extends Controller\n{\n"
            "    public function show(Request $request)\n    {\n"
            "        return view('profile', ['name' => $request->user()->getFullName()]);\n    }\n}\n"
        ),
        "resources/views/layouts/app.blade.php": (
            "<!DOCTYPE html>\n<html>\n<head><title>@yield('title', 'Shop')</title>\n@vite(['resources/css/app.css'])\n</head>\n"
            "<body>\n<nav><a href=\"{{ route('home') }}\">Home</a> <a href=\"{{ route('orders.index') }}\">Orders</a></nav>\n"
            "<main>@yield('content')</main>\n</body>\n</html>\n"
        ),
        "resources/views/home.blade.php": "@extends('layouts.app')\n\n@section('content')\n<h1>Welcome</h1>\n@endsection\n",
        "resources/views/orders/index.blade.php": (
            "@extends('layouts.app')\n\n@section('content')\n<h1>Your orders</h1>\n"
            "@foreach ($orders as $order)\n<p>#{{ $order->id }} {{ $order->total }}</p>\n@endforeach\n@endsection\n"
        ),
        "resources/views/profile.blade.php": "@extends('layouts.app')\n\n@section('content')\n<h1>{{ $name }}</h1>\n@endsection\n",
        "resources/views/emails/welcome.blade.php": "<p>Hello {{ $user->getFullName() }}, welcome!</p>\n",
    }
    # Unrelated files so the workspace is not trivially small.
    for i in range(padding):
        files[f"app/Support/Helpers/Helper{i:02d}.php"] = (
            f"<?php\n\nnamespace App\\Support\\Helpers;\n\nclass Helper{i:02d}\n{{\n"
            f"    public static function value(): int\n    {{\n        return {i};\n    }}\n}}\n"
        )
        files[f"resources/views/partials/block{i:02d}.blade.php"] = f"<div class=\"block-{i}\">Block {i}</div>\n"
    write_tree(ws, files)


def hypr_skeleton(ws: Path) -> None:
    write_tree(ws, {
        "hypr/hyprland.conf": (
            "# Main config\nsource = ~/.config/hypr/conf/monitors.conf\nsource = ~/.config/hypr/conf/general.conf\n"
            "source = ~/.config/hypr/conf/keybinds.conf\nsource = ~/.config/hypr/conf/autostart.conf\n"
        ),
        "hypr/conf/monitors.conf": "monitor = DP-1, 2560x1440@165, 0x0, 1\nmonitor = DP-2, 2560x1440@144, 2560x0, 1\n",
        "hypr/conf/general.conf": (
            "general {\n    gaps_in = 5\n    gaps_out = 10\n    border_size = 2\n"
            "    col.active_border = rgba(89b4faee)\n    layout = dwindle\n}\n\n"
            "decoration {\n    rounding = 8\n}\n"
        ),
        "hypr/conf/keybinds.conf": (
            "$mainMod = SUPER\n$terminal = ghostty\n$fileManager = thunar\n\n"
            "bind = $mainMod, Return, exec, $terminal\n"
            "bind = $mainMod, E, exec, $fileManager\n"
            "bind = $mainMod, Q, killactive,\n"
            "bind = $mainMod SHIFT, F, togglefloating,\n"
            "bind = $mainMod, 1, workspace, 1\nbind = $mainMod, 2, workspace, 2\n"
        ),
        "hypr/conf/autostart.conf": "exec-once = waybar\nexec-once = swaync\n",
        "waybar/config.jsonc": '{\n  "layer": "top",\n  "modules-left": ["hyprland/workspaces"],\n  "modules-right": ["clock", "pulseaudio"]\n}\n',
        "waybar/style.css": "* { font-family: JetBrainsMono; }\n",
        "README.md": "# hyprdots\nMy Hyprland dotfiles.\n",
    })


def cart_project(ws: Path) -> None:
    write_tree(ws, {
        "src/Cart.php": (
            "<?php\n\nclass Cart\n{\n    private array $items = [];\n\n"
            "    public function add(string $sku, float $price, int $qty = 1): void\n    {\n"
            "        $this->items[$sku] = ['price' => $price, 'qty' => $qty];\n    }\n\n"
            "    public function total(float $discountPercent = 0): float\n    {\n        $sum = 0;\n"
            "        foreach ($this->items as $item) {\n            $sum += $item['price'];\n        }\n\n"
            "        return round($sum - $sum * $discountPercent, 2);\n    }\n}\n"
        ),
        "tests/run.php": (
            "<?php\nrequire __DIR__.'/../src/Cart.php';\n\n$fail = 0;\n"
            "function check($name, $got, $want) { global $fail; if (abs($got - $want) > 0.001) { echo \"FAIL $name: got $got want $want\\n\"; $fail++; } else { echo \"ok $name\\n\"; } }\n\n"
            "$c = new Cart();\n$c->add('a', 10.0, 2);\n$c->add('b', 5.5);\ncheck('total', $c->total(), 25.5);\n"
            "check('discount 10%', $c->total(10), 22.95);\n\n"
            "$d = new Cart();\n$d->add('a', 3.0);\n$d->add('a', 3.0, 2);\ncheck('same sku adds up', $d->total(), 9.0);\n\n"
            "exit($fail ? 1 : 0);\n"
        ),
    })


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    check: Callable[[Path, str], str | None]  # returns a failure reason, or None on success
    notes: str = ""


def check_find_tax(ws: Path, reply: str) -> str | None:
    if "app/Services/Billing/InvoiceCalculator.php" not in reply.replace("\\", "/"):
        return "reply does not name app/Services/Billing/InvoiceCalculator.php"
    if not re.search(r"0?\.19\b|19\s?%", reply):
        return "reply does not give the German rate (0.19 / 19%)"
    return None


def check_cart(ws: Path, reply: str) -> str | None:
    result = subprocess.run(["php", "tests/run.php"], cwd=ws, capture_output=True, text=True)
    if result.returncode != 0:
        return "tests still fail: " + result.stdout.strip().replace("\n", " | ")[-300:]
    original = Path(ws / ".orig_tests").read_text() if (ws / ".orig_tests").exists() else None
    if original is not None and (ws / "tests/run.php").read_text() != original:
        return "tests/run.php was modified"
    return None


def setup_cart(ws: Path) -> None:
    cart_project(ws)
    (ws / ".orig_tests").write_text((ws / "tests/run.php").read_text())


def check_rename(ws: Path, reply: str) -> str | None:
    leftovers = [
        str(p.relative_to(ws)) for p in ws.rglob("*")
        if p.is_file() and "getFullName" in p.read_text(errors="ignore")
    ]
    if leftovers:
        return "getFullName still used in: " + ", ".join(leftovers)
    user = (ws / "app/Models/User.php").read_text()
    if "function fullName(" not in user:
        return "User::fullName() not defined"
    for rel in ("app/Mail/OrderShipped.php", "app/Http/Controllers/ProfileController.php"):
        if "fullName()" not in (ws / rel).read_text():
            return f"{rel} does not call fullName()"
    if "fullName()" not in (ws / "resources/views/emails/welcome.blade.php").read_text():
        return "welcome.blade.php does not call fullName()"
    return php_lint(ws, "app/Models/User.php", "app/Mail/OrderShipped.php",
                    "app/Http/Controllers/ProfileController.php")


def check_dashboard(ws: Path, reply: str) -> str | None:
    routes = (ws / "routes/web.php").read_text()
    if "DashboardController" not in routes or "'/dashboard'" not in routes and '"/dashboard"' not in routes:
        return "routes/web.php has no /dashboard route to DashboardController"
    # The route must sit inside the auth group or carry ->middleware('auth').
    group = re.search(r"Route::middleware\('auth'\)->group\(function \(\) \{(.*?)\n\}\);", routes, re.S)
    in_group = bool(group and "dashboard" in group.group(1))
    if not in_group and not re.search(r"dashboard.*middleware\(\s*['\"]auth", routes, re.S):
        return "/dashboard is not protected by the auth middleware"
    if "->name('dashboard')" not in routes and '->name("dashboard")' not in routes:
        return "route is not named 'dashboard'"
    controller = ws / "app/Http/Controllers/DashboardController.php"
    if not controller.exists():
        return "app/Http/Controllers/DashboardController.php missing"
    text = controller.read_text()
    if "namespace App\\Http\\Controllers;" not in text or "function index" not in text:
        return "DashboardController lacks the namespace or index()"
    if not re.search(r"view\(\s*['\"]dashboard['\"]", text):
        return "index() does not return view('dashboard')"
    view = ws / "resources/views/dashboard.blade.php"
    if not view.exists():
        return "resources/views/dashboard.blade.php missing"
    if "@extends('layouts.app')" not in view.read_text():
        return "dashboard view does not extend layouts.app"
    if "route('dashboard')" not in (ws / "resources/views/layouts/app.blade.php").read_text():
        return "layout nav has no link to route('dashboard')"
    if (ws / "app/Http/Kernel.php").exists():
        return "created app/Http/Kernel.php (removed in Laravel 11+)"
    return php_lint(ws, "routes/web.php", "app/Http/Controllers/DashboardController.php")


def check_hypr(ws: Path, reply: str) -> str | None:
    binds = (ws / "hypr/conf/keybinds.conf").read_text()
    if not re.search(r"^bind\s*=\s*\$mainMod\s+SHIFT\s*,\s*B\s*,\s*exec\s*,\s*zen-browser\s*$", binds, re.M | re.I):
        return "keybinds.conf lacks `bind = $mainMod SHIFT, B, exec, zen-browser`"
    if binds.count("bind") != 7:
        return f"keybinds.conf has {binds.count('bind')} binds, expected 7 (6 old + 1 new)"
    general = (ws / "hypr/conf/general.conf").read_text()
    if not re.search(r"^\s*gaps_in\s*=\s*4\s*$", general, re.M):
        return "general.conf gaps_in is not 4"
    if "gaps_out = 10" not in general or "rounding = 8" not in general:
        return "general.conf lost other settings"
    if "gaps_in" in (ws / "hypr/hyprland.conf").read_text():
        return "gaps_in was added to hyprland.conf instead of conf/general.conf"
    return None


TASKS = [
    Task(
        "find_tax",
        "Where is calculateInvoiceTax defined in this project, and what tax rate does it apply "
        "for the region 'de'? Give the file path relative to the workspace.",
        lambda ws: laravel_skeleton(ws, padding=120),
        check_find_tax,
        "read-only navigation in a ~260-file workspace",
    ),
    Task(
        "cart_bugfix",
        "The tests in tests/run.php fail. Fix the bugs in src/Cart.php so that `php tests/run.php` "
        "passes. Do not change the tests. Run the tests to confirm.",
        setup_cart,
        check_cart,
        "two bugs (qty ignored, same sku overwrites, percent not divided)",
    ),
    Task(
        "rename_method",
        "Rename the User model method getFullName() to fullName() and update every caller in the "
        "project, including Blade views.",
        lambda ws: laravel_skeleton(ws, padding=60),
        check_rename,
        "multi-file refactor incl. Blade",
    ),
    Task(
        "laravel_dashboard",
        "This is a Laravel 12 app. Add an authenticated dashboard: a DashboardController with an "
        "index action returning a `dashboard` view, a GET /dashboard route named `dashboard` that "
        "requires login (use the existing auth route group), a resources/views/dashboard.blade.php "
        "view that extends the app layout with a heading 'Dashboard', and a Dashboard link in the "
        "layout nav. Follow the project's existing conventions.",
        lambda ws: laravel_skeleton(ws, padding=60),
        check_dashboard,
        "multi-file feature following conventions",
    ),
    Task(
        "hypr_keybind",
        "In my Hyprland dotfiles here, add a keybind SUPER+SHIFT+B that launches zen-browser, "
        "following the existing keybind style, and change the inner gaps to 4. Edit the files "
        "where those settings already live.",
        hypr_skeleton,
        check_hypr,
        "dotfiles: find the sourced file, match conventions",
    ),
]


# ---------------------------------------------------------------------------
# ACP client
# ---------------------------------------------------------------------------

class EvalClient:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.permissions = 0

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.permissions += 1
        ids = [o.option_id for o in options]
        choice = "approve_once" if "approve_once" in ids else ids[0]
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=choice))

    async def session_update(self, session_id, update, **kwargs):
        if getattr(update, "session_update", None) == "agent_message_chunk":
            content = getattr(update, "content", None)
            if getattr(content, "text", None):
                self.text.append(content.text)

    async def write_text_file(self, *a, **k):
        raise NotImplementedError

    async def read_text_file(self, *a, **k):
        raise NotImplementedError

    def on_connect(self, conn) -> None:
        pass


def read_log(path: Path) -> dict[str, Any]:
    steps = tool_calls = failed = 0
    prompt_tokens: list[int] = []
    reply_tokens = 0
    prompt_seconds = reply_seconds = 0.0
    tools: list[str] = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "model":
                steps += 1
                if event.get("prompt_tokens"):
                    prompt_tokens.append(event["prompt_tokens"])
                reply_tokens += event.get("reply_tokens") or 0
                prompt_seconds += event.get("prompt_seconds") or 0
                reply_seconds += event.get("reply_seconds") or 0
            elif event.get("event") == "tool":
                tool_calls += 1
                tools.append(event.get("tool", "?"))
                result = str(event.get("result", ""))
                if result.lstrip().startswith(("Rejected", "Invalid", "Unknown tool", "Edit failed", "Tool failed", "Not found", "Blocked")):
                    failed += 1
    return {
        "steps": steps, "tool_calls": tool_calls, "failed_tool_calls": failed,
        "max_prompt_tokens": max(prompt_tokens, default=0), "reply_tokens": reply_tokens,
        "prompt_seconds": round(prompt_seconds, 1), "reply_seconds": round(reply_seconds, 1),
        "tools": tools,
    }


async def run_task(task: Task, env_overrides: dict[str, str]) -> dict[str, Any]:
    base = Path(tempfile.mkdtemp(prefix=f"agent-eval-{task.name}-"))
    ws = base / "ws"
    ws.mkdir()
    task.setup(ws)
    state = base / "state"
    log_path = base / "session.jsonl"
    env = {
        **os.environ,
        "OLLAMA_AGENT_STATE_DIR": str(state),
        "OLLAMA_AGENT_SESSION_LOG": str(log_path),
        "OLLAMA_AGENT_SESSION_APPROVALS": "0",
        **env_overrides,
    }
    client = EvalClient()
    started = time.time()
    stop_reason = error = None
    try:
        async with spawn_agent_process(lambda agent: client, sys.executable, str(AGENT), env=env, cwd=str(ws)) as (conn, _proc):
            await conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities())
            session = await conn.new_session(cwd=str(ws), mcp_servers=[])
            response = await asyncio.wait_for(
                conn.prompt(session_id=session.session_id, prompt=[text_block(task.prompt)]),
                timeout=TASK_TIMEOUT,
            )
            stop_reason = response.stop_reason
    except asyncio.TimeoutError:
        error = f"timed out after {TASK_TIMEOUT}s"
    except Exception as exc:  # noqa: BLE001 - report every failure
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started
    reply = "".join(client.text)
    failure = error or task.check(ws, reply)
    result = {
        "task": task.name, "passed": failure is None, "failure": failure,
        "seconds": round(elapsed, 1), "stop_reason": stop_reason,
        "permissions": client.permissions, **read_log(log_path),
        "reply_tail": reply[-600:], "workspace": str(ws),
    }
    if failure is None and os.environ.get("EVAL_KEEP") != "1":
        shutil.rmtree(base, ignore_errors=True)
    return result


async def main(selected: list[str]) -> int:
    model = os.environ.get("OLLAMA_AGENT_MODEL", "default")
    tasks = [t for t in TASKS if not selected or t.name in selected]
    results = []
    for task in tasks:
        print(f"--> {task.name}: {task.notes}", flush=True)
        result = await run_task(task, {})
        results.append(result)
        status = "PASS" if result["passed"] else f"FAIL ({result['failure']})"
        print(f"    {status} in {result['seconds']}s, {result['steps']} steps, "
              f"{result['tool_calls']} tools ({result['failed_tool_calls']} failed), "
              f"max prompt {result['max_prompt_tokens']} tok, model time "
              f"{result['prompt_seconds']}s reading + {result['reply_seconds']}s writing "
              f"({result['reply_tokens']} tok)", flush=True)
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed, {sum(r['seconds'] for r in results):.0f}s total")
    out = ROOT / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{model.replace('/', '_').replace(':', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out / name).write_text(json.dumps({"model": model, "results": results}, indent=2))
    print(f"Details: {out / name}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))

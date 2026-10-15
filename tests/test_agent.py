"""Self-tests for hypr_agent_zed.py (standard library only).

Run with the agent's interpreter so the `acp` package is importable:
    ~/.venvs/zed-agent/bin/python -m unittest discover -s tests -v
"""

import http.server
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hypr_agent_zed as h  # noqa: E402


def _workspace() -> Path:
    # Under /tmp, never under $HOME: sandboxed runs refuse a workspace containing home.
    return Path(tempfile.mkdtemp(prefix="agent-test-")).resolve()


class EditFileTests(unittest.TestCase):
    def setUp(self):
        self.ws = _workspace()
        self.file = self.ws / "a.py"
        self.file.write_text('class A:\n    KEEP = os.environ.get("X", "30m")\n    y = 1\n')

    def edit(self, old, new):
        return h.edit_file("a.py", old, new, cwd=str(self.ws))

    def test_exact(self):
        self.assertTrue(self.edit("y = 1", "y = 2").startswith("Edited"))
        self.assertIn("y = 2", self.file.read_text())

    def test_json_escaped_quotes(self):
        result = self.edit('KEEP = os.environ.get(\\"X\\", \\"30m\\")', 'KEEP = os.environ.get("X", "30s")')
        self.assertTrue(result.startswith("Edited"), result)
        self.assertIn('"30s"', self.file.read_text())

    def test_wrong_indentation_is_reindented(self):
        result = self.edit('KEEP = os.environ.get("X", "30m")', 'KEEP = 5')
        self.assertTrue(result.startswith("Edited"), result)
        self.assertEqual(self.file.read_text(), "class A:\n    KEEP = 5\n    y = 1\n")

    def test_read_file_line_numbers_are_stripped(self):
        result = self.edit('2:     KEEP = os.environ.get("X", "30m")\n3:     y = 1',
                           '2:     KEEP = 5\n3:     y = 2')
        self.assertTrue(result.startswith("Edited"), result)
        self.assertEqual(self.file.read_text(), "class A:\n    KEEP = 5\n    y = 2\n")

    def test_line_numbers_in_old_only(self):
        result = self.edit("3:     y = 1", "    y = 2")
        self.assertTrue(result.startswith("Edited"), result)
        self.assertIn("    y = 2\n", self.file.read_text())

    def test_missing_shows_closest_line(self):
        result = self.edit('KEEP = os.environ.get("Z", "30m")', "q")
        self.assertIn("not found", result)
        self.assertIn("Closest line", result)

    def test_ambiguous(self):
        self.file.write_text("a = 1\na = 1\n")
        self.assertIn("matched 2 times", self.edit("a = 1", "a = 2"))

    def test_outside_workspace_rejected(self):
        other = _workspace() / "b.txt"
        other.write_text("x")
        self.assertIn("Rejected", h.edit_file(str(other), "x", "y", cwd=str(self.ws)))
        self.assertEqual(other.read_text(), "x")

    def test_git_and_sensitive_paths_rejected(self):
        (self.ws / ".git").mkdir()
        (self.ws / ".git" / "config").write_text("x")
        self.assertIn("Rejected", h.edit_file(".git/config", "x", "y", cwd=str(self.ws)))
        (self.ws / ".env").write_text("K=1")
        self.assertIn("Rejected", h.edit_file(".env", "K=1", "K=2", cwd=str(self.ws)))


class EditPreviewTests(unittest.TestCase):
    def setUp(self):
        self.ws = _workspace()
        (self.ws / "a.py").write_text("x = 1\ny = 2\n")

    def test_prepare_edit_gives_full_before_and_after_without_writing(self):
        error, planned = h._prepare_edit("a.py", "y = 2", "y = 3", cwd=str(self.ws))
        self.assertIsNone(error)
        path, before, after, _ = planned
        self.assertEqual((before, after), ("x = 1\ny = 2\n", "x = 1\ny = 3\n"))
        self.assertEqual((self.ws / "a.py").read_text(), "x = 1\ny = 2\n")

    def test_prepare_edit_reports_errors(self):
        error, planned = h._prepare_edit("a.py", "nope", "z", cwd=str(self.ws))
        self.assertIsNone(planned)
        self.assertIn("not found", error)

    def test_agent_builds_diff_for_edit_and_new_file(self):
        agent = h.OllamaZedAgent()
        agent.session_cwds["s"] = str(self.ws)
        edit = agent._file_diff("s", "edit_file", {"path": "a.py", "old": "y = 2", "new": "y = 3"})
        self.assertEqual((edit[0].old_text, edit[0].new_text), ("x = 1\ny = 2\n", "x = 1\ny = 3\n"))
        new = agent._file_diff("s", "write_file", {"path": "n.txt", "content": "hi"})
        self.assertIsNone(new[0].old_text)
        self.assertEqual(new[0].new_text, "hi")
        self.assertIsNone(agent._file_diff("s", "edit_file", {"path": "a.py", "old": "nope", "new": "z"}))


class SessionLogTests(unittest.TestCase):
    def setUp(self):
        self._path = h.SESSION_LOG
        h.SESSION_LOG = _workspace() / "session.jsonl"

    def tearDown(self):
        h.SESSION_LOG = self._path

    def test_reset_empties_and_events_append(self):
        h.log_event("user", text="old")
        h.reset_session_log()
        self.assertEqual(h.SESSION_LOG.read_text(), "")
        h.log_event("tool", tool="edit_file", result="x" * (h.SESSION_LOG_MAX_CHARS + 10))
        entry = json.loads(h.SESSION_LOG.read_text())
        self.assertEqual(entry["event"], "tool")
        self.assertIn("more chars", entry["result"])
        self.assertEqual(oct(h.SESSION_LOG.stat().st_mode & 0o777), "0o600")


class WriteFileTests(unittest.TestCase):
    def setUp(self):
        self.ws = _workspace()

    def test_create_and_no_overwrite(self):
        self.assertTrue(h.write_file("sub/n.txt", "hi", cwd=str(self.ws)).startswith("Created"))
        self.assertIn("already exists", h.write_file("sub/n.txt", "x", cwd=str(self.ws)))

    def test_home_is_out_of_scope_by_default(self):
        self.assertFalse(h.WRITE_HOME)
        target = h.HOME / "agent_test_should_not_exist.txt"
        self.assertIn("Rejected", h.write_file(str(target), "x", cwd=str(self.ws)))
        self.assertFalse(target.exists())

    def test_symlink_rejected(self):
        (self.ws / "link").symlink_to("/etc")
        self.assertIn("Rejected", h.write_file("link/x.txt", "x", cwd=str(self.ws)))


class ArgumentTests(unittest.TestCase):
    def test_aliases_drops_and_coercion(self):
        args, notes = h.normalize_tool_arguments(
            "read_file", {"file_path": "a.py", "start_line": "3", "reason": "why", "line_end": None}
        )
        self.assertEqual(args, {"path": "a.py", "line_start": 3})
        self.assertEqual(len(notes), 3)
        self.assertIsNone(h.validate_tool_arguments("read_file", args))

    def test_bool_and_env_coercion(self):
        args, _ = h.normalize_tool_arguments(
            "run_tests", {"failed": "true", "env": '{"APP_ENV": "test"}'}
        )
        self.assertEqual(args, {"failed_only": True, "env": {"APP_ENV": "test"}})
        self.assertIsNone(h.validate_tool_arguments("run_tests", args))

    def test_unknown_tool_untouched(self):
        self.assertEqual(h.normalize_tool_arguments("nope", {"a": 1}), ({"a": 1}, []))

    def test_validation_still_rejects_garbage(self):
        self.assertIsNotNone(h.validate_tool_arguments("edit_file", {"path": "a"}))
        self.assertIsNotNone(h.validate_tool_arguments("read_file", {"path": "a", "line_start": 0}))
        self.assertIsNotNone(h.validate_tool_arguments("setup_test_env", {"kind": "ruby"}))

    def test_env_policy(self):
        self.assertIsNone(h._env_problem({"APP_ENV": "test", "N": 1}))
        for bad in ({"PATH": "/x"}, {"LD_PRELOAD": "x"}, {"API_KEY": "x"}, {"lower": "x"}, {"NODE_OPTIONS": "x"}):
            self.assertIsNotNone(h._env_problem(bad), bad)


class CommandPolicyTests(unittest.TestCase):
    def test_validate_command(self):
        for bad in ("sudo ls", "ls | wc", "cat ~/.ssh/id_rsa", "cat .env", "rm -rf /; ls", "curl x"):
            argv, err = h.validate_command(bad)
            self.assertIsNone(argv, bad)
            self.assertTrue(err)

    def test_read_only_needs_no_approval_but_mutations_do(self):
        self.assertFalse(h.command_needs_approval("ls -la"))
        self.assertFalse(h.command_needs_approval("git status"))
        self.assertTrue(h.command_needs_approval("git commit -m x"))
        self.assertTrue(h.command_needs_approval("sed -i s/a/b/ f"))
        self.assertTrue(h.command_needs_approval("npm install"))

    def test_test_command_policy(self):
        ws = _workspace()
        argv, err = h.validate_test_command("python3 -m pytest -q", ws)
        self.assertIsNone(err)
        for bad in (
            "python3 -c print(1)", "ruff check --fix .", "black .", "pytest; ls",
            "sudo pytest", "python3 /etc/x.py", "make -f /tmp/x test",
        ):
            argv, err = h.validate_test_command(bad, ws)
            self.assertIsNone(argv, bad)


class GitPolicyTests(unittest.TestCase):
    def test_everything_useful_is_allowed_with_approval(self):
        for cmd in (
            "git push origin main", "git pull --rebase", "git fetch --all", "git merge dev --no-edit",
            "git rebase main", "git reset --hard HEAD~1", "git revert abc123", "git cherry-pick -x abc",
            "git clean -fd", "git switch -c feature", "git checkout -b x", "git stash pop",
            "git commit -c abc123", "git remote add origin git@github.com:a/b.git",
            "git config user.name robb", "git submodule update --init", "git branch -D old",
        ):
            argv, err = h.validate_command(cmd)
            self.assertIsNone(err, cmd)
            self.assertTrue(h.command_needs_approval(cmd), cmd)

    def test_read_only_queries_need_no_approval(self):
        for cmd in ("git status", "git log -n 5", "git blame a.py", "git --no-pager diff", "git grep foo"):
            self.assertFalse(h.command_needs_approval(cmd), cmd)

    def test_code_running_forms_are_refused(self):
        for cmd in (
            "git -c core.pager=sh log", "git -C /etc status", "git --git-dir=/x status",
            "git filter-branch --tree-filter x", "git difftool", "git rebase -x make main",
            "git rebase --exec make main", "git bisect run make", "git submodule foreach ls",
            "git clone -c core.sshCommand=x url", "git clone -u x url",
            "git remote add o ext::sh", "git push --receive-pack=x origin",
            "git config core.sshCommand x", "git config alias.x !sh", "git config --global user.name x",
            "git config credential.helper store", "git grep -Osh foo", "git log --output=x",
            "git co", "git",
        ):
            argv, err = h.validate_command(cmd)
            self.assertIsNone(argv, cmd)

    def test_remote_detection_and_session_approval(self):
        self.assertTrue(h.git_is_remote(["push", "origin"]))
        self.assertTrue(h.git_is_remote(["remote", "update"]))
        self.assertFalse(h.git_is_remote(["remote", "add", "o", "u"]))
        self.assertFalse(h.git_is_remote(["commit", "-m", "x"]))
        self.assertTrue(h.git_session_approvable(["commit", "-m", "x"]))
        for args in (["push"], ["reset", "--hard"], ["commit", "--amend"], ["clean", "-fd"], ["checkout", "."]):
            self.assertFalse(h.git_session_approvable(args), args)

    def test_paths_outside_workspace_refused(self):
        ws = _workspace()
        self.assertIn("outside", h.git_path_problem(["clone", "https://x/y", "/tmp/elsewhere"], ws))
        self.assertIn("outside", h.git_path_problem(["format-patch", "-o", "../x", "HEAD~1"], ws))
        self.assertIsNone(h.git_path_problem(["remote", "add", "backup", "/mnt/backup/r.git"], ws))
        self.assertIsNone(h.git_path_problem(["clone", "/mnt/src/repo", "sub"], ws))
        self.assertIn("outside", h.git_path_problem(["archive", "--prefix=p/", "-o", "../a.tar", "HEAD"], ws))
        self.assertIsNone(h.git_path_problem(["clone", "git@github.com:a/b.git", "sub"], ws))
        self.assertIsNone(h.git_path_problem(["add", "src/a.py"], ws))
        self.assertIsNone(h.git_path_problem(["log", "main..origin/main"], ws))


@unittest.skipUnless(shutil.which("git"), "git not installed")
class GitEndToEndTests(unittest.TestCase):
    def test_commit_push_pull_and_reset(self):
        root = _workspace()
        remote, ws, other = root / "remote.git", root / "ws", root / "other"
        ws.mkdir()
        other.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t"}
        old_env = dict(os.environ)
        os.environ.update(env)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(old_env)))
        import subprocess
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)

        def run(cwd, command):
            return h.run_command(command, cwd=str(cwd))

        self.assertNotIn("Rejected", run(ws, "git init -b main"))
        (ws / "a.txt").write_text("one\n")
        run(ws, "git add a.txt")
        self.assertIn("a.txt", run(ws, 'git commit -m first'))
        run(ws, f"git remote add origin {remote}")
        pushed = run(ws, "git push -u origin main")
        self.assertNotIn("Rejected", pushed)
        self.assertIn("first", subprocess.run(["git", "--git-dir", str(remote), "log", "--oneline"],
                                              capture_output=True, text=True).stdout)
        # another clone pulls it, then the workspace resets back
        subprocess.run(["git", "clone", "-q", str(remote), str(other / "c")], check=True)
        (ws / "a.txt").write_text("two\n")
        run(ws, "git commit -am second")
        run(ws, "git push")
        self.assertNotIn("Rejected", run(other / "c", "git pull"))
        self.assertEqual((other / "c" / "a.txt").read_text(), "two\n")
        run(ws, "git reset --hard HEAD~1")
        self.assertEqual((ws / "a.txt").read_text(), "one\n")


class SetupEnvTests(unittest.TestCase):
    def setUp(self):
        self.ws = _workspace()
        self._mode = h.SANDBOX_MODE
        h.SANDBOX_MODE = "off"  # rejection paths return before anything runs

    def tearDown(self):
        h.SANDBOX_MODE = self._mode

    def test_rejects_urls_paths_and_options(self):
        for spec in ("git+https://x/y", "../evil", "--index-url=http://x", "a;b", "/tmp/x.whl"):
            result = h.setup_test_env(".", "python", spec, cwd=str(self.ws))
            self.assertIn("Rejected", result, spec)
        self.assertFalse((self.ws / ".venv").exists())

    def test_directory_must_be_inside_workspace(self):
        self.assertIn("Rejected", h.setup_test_env("..", "python", None, cwd=str(self.ws)))
        self.assertIn("Rejected", h.setup_test_env("missing", "python", None, cwd=str(self.ws)))

    def test_node_needs_package_json(self):
        self.assertIn("no package.json", h.setup_test_env(".", "node", None, cwd=str(self.ws)))

    def test_package_spec_regex(self):
        for good in ("pytest", "requests>=2.0", "pkg[extra]==1.2.3", "a_b-c.d"):
            self.assertTrue(h._PACKAGE_SPEC.fullmatch(good), good)


@unittest.skipUnless(h.sandbox_available(), "bubblewrap sandbox not available")
class RunTestsTests(unittest.TestCase):
    def setUp(self):
        self.ws = _workspace()
        (self.ws / "test_demo.py").write_text(
            "import unittest, pathlib, os\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        pathlib.Path('leak.txt').write_text('x')\n"
            "    def test_env(self):\n"
            "        self.assertEqual(os.environ.get('APP_ENV'), 'test')\n"
        )

    def run_tests(self, **kw):
        return h.run_tests("python3 -m unittest -q", cwd=str(self.ws), **kw)

    def test_writes_are_discarded_and_env_is_passed(self):
        failing = self.run_tests()
        self.assertIn("FAILED", failing)
        passing = self.run_tests(env_vars={"APP_ENV": "test"})
        self.assertIn("PASSED", passing, passing)
        self.assertFalse((self.ws / "leak.txt").exists(), "test run changed the workspace")

    def test_network_blocked(self):
        (self.ws / "net_probe.py").write_text(
            "import socket,sys\n"
            "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    sys.exit(0)\n"
            "except OSError:\n    sys.exit(1)\n"
        )
        self.assertIn("FAILED", h.run_tests("python3 net_probe.py", cwd=str(self.ws)))

    def test_directory_escape_rejected(self):
        self.assertIn("Rejected", h.run_tests("python3 -m unittest", directory="..", cwd=str(self.ws)))

    def test_failed_only_without_history(self):
        self.assertIn("Rejected", h.run_tests(failed_only=True, cwd=str(_workspace())))

    def test_nested_project_hint(self):
        ws = _workspace()
        (ws / "app").mkdir()
        (ws / "app" / "pyproject.toml").write_text("")
        self.assertIn("sub-folders: app", h.run_tests(cwd=str(ws)))

    def test_secrets_not_leaked_into_sandbox(self):
        os.environ["MY_API_KEY"] = "topsecret"
        try:
            (self.ws / "env_probe.py").write_text(
                "import os,sys\nsys.exit(1 if 'MY_API_KEY' in os.environ else 0)\n"
            )
            self.assertIn("PASSED", h.run_tests("python3 env_probe.py", cwd=str(self.ws)))
        finally:
            del os.environ["MY_API_KEY"]


class SummaryTests(unittest.TestCase):
    def test_pytest_failures_are_extracted(self):
        digest, failing = h.summarise_test_output(
            "FAILED tests/test_a.py::test_b - assert 1 == 2\n=== 1 failed, 3 passed in 0.12s ===\n"
        )
        self.assertEqual(failing, ["FAILED tests/test_a.py::test_b"])
        self.assertTrue(any("1 failed" in line for line in digest))

    def test_failure_detection(self):
        self.assertTrue(h.is_failed_result("Rejected: x"))
        self.assertTrue(h.is_failed_result("Invalid tool call for x"))
        self.assertFalse(h.is_failed_result("Edited /a/b."))


class ModelOptionTests(unittest.TestCase):
    def test_qwen3_gets_sampling_settings(self):
        old = h.MODEL
        try:
            h.MODEL = "qwen3:14b"
            options = h.model_options()
            self.assertEqual(options["temperature"], 0.6)
            self.assertEqual(options["top_k"], 20)
            h.MODEL = "gpt-oss:20b"
            self.assertNotIn("temperature", h.model_options())
        finally:
            h.MODEL = old


class FakeOllama(http.server.BaseHTTPRequestHandler):
    chunks: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write((json.dumps(chunk) + "\n").encode())

    def log_message(self, *args):
        pass


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._url = h.OLLAMA_URL
        h.OLLAMA_URL = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        h.OLLAMA_URL = self._url
        self.server.shutdown()
        self.server.server_close()

    def collect(self, chunks, budget=0):
        FakeOllama.chunks = chunks
        events = queue.Queue()
        h.ollama_stream_request("/api/chat", {}, events, think_budget=budget)
        out = []
        while not events.empty():
            out.append(events.get())
        return out

    def test_tool_calls_in_separate_chunks(self):
        def call(name, **a):
            return {"message": {"tool_calls": [{"function": {"name": name, "arguments": a}}]}}
        events = self.collect([call("read_file", path="a"), call("list_files", path="."), {"done": True}])
        done = [v for e, v in events if e == "done"][0]
        names = [c["function"]["name"] for c in done["message"]["tool_calls"]]
        self.assertEqual(names, ["read_file", "list_files"])

    def test_thinking_streams_and_budget_overflow(self):
        think = {"message": {"thinking": "x" * 100}}
        events = self.collect([think, think, {"message": {"content": "hi"}}], budget=150)
        kinds = [e for e, _ in events]
        self.assertIn("think_overflow", kinds)
        self.assertNotIn("done", kinds)
        events = self.collect([think, think, {"message": {"content": "hi"}}], budget=0)
        kinds = [e for e, _ in events]
        self.assertEqual(kinds.count("thinking"), 2)
        self.assertEqual(kinds[-1], "done")


if __name__ == "__main__":
    unittest.main()

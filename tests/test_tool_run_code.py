import sys

import pytest

from bot.agent.tools.base import Tier, ToolContext, ToolError
from bot.agent.tools.run_code import (
    MAX_OUTPUT_CHARS,
    CommandResult,
    SandboxConfig,
    run_code_tool,
    run_command,
)

CTX = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)
CONFIG = SandboxConfig(image="images:alpine/3.20", profile="sandbox", timeout=12.0)
OK = CommandResult(0, "", "")


class FakeRunner:
    """Records every ``incus`` invocation and answers by subcommand."""

    def __init__(
        self,
        *,
        launch: CommandResult = OK,
        launch_error: BaseException | None = None,
        exec_result: CommandResult = OK,
        exec_error: BaseException | None = None,
        delete_error: BaseException | None = None,
    ) -> None:
        self.launch = launch
        self.launch_error = launch_error
        self.exec_result = exec_result
        self.exec_error = exec_error
        self.delete_error = delete_error
        self.calls: list[tuple[list[str], bytes | None, float]] = []

    async def __call__(self, argv: list[str], stdin: bytes | None, timeout: float) -> CommandResult:
        self.calls.append((argv, stdin, timeout))
        match argv[1]:
            case "launch":
                if self.launch_error is not None:
                    raise self.launch_error
                return self.launch
            case "exec":
                if self.exec_error is not None:
                    raise self.exec_error
                return self.exec_result
            case "delete":
                if self.delete_error is not None:
                    raise self.delete_error
                return OK
        raise AssertionError(f"unexpected command: {argv}")

    def argv(self, subcommand: str) -> list[list[str]]:
        return [argv for argv, _, _ in self.calls if argv[1] == subcommand]


def _option_values(argv: list[str], option: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == option]


def test_tool_metadata() -> None:
    tool = run_code_tool(CONFIG, runner=FakeRunner())

    assert tool.name == "run_code"
    assert tool.tier is Tier.SANDBOX
    assert "12s" in tool.description
    parameters = tool.spec()["function"]["parameters"]
    assert parameters["required"] == ["code"]
    assert parameters["properties"]["language"]["enum"] == ["python", "sh"]


async def test_launches_an_ephemeral_limited_container() -> None:
    runner = FakeRunner()
    config = SandboxConfig(image="img", profile="nonet", memory="128MiB", cpu="2", processes="32")

    await run_code_tool(config, runner=runner).run({"code": "print(1)"}, CTX)

    (launch,) = runner.argv("launch")
    assert launch[:3] == ["incus", "launch", "img"]
    assert launch[3].startswith("bot-sandbox-")
    assert "--ephemeral" in launch
    assert _option_values(launch, "--profile") == ["nonet"]
    configs = _option_values(launch, "--config")
    assert "limits.memory=128MiB" in configs
    assert "limits.cpu=2" in configs
    assert "limits.processes=32" in configs
    assert "security.privileged=false" in configs


async def test_code_is_piped_to_python_in_the_same_container() -> None:
    runner = FakeRunner()
    code = "print('héllo 你好')"

    await run_code_tool(CONFIG, runner=runner).run({"code": code}, CTX)

    name = runner.argv("launch")[0][3]
    exec_argv, stdin, timeout = next(call for call in runner.calls if call[0][1] == "exec")
    assert exec_argv[:3] == ["incus", "exec", name]
    assert exec_argv[exec_argv.index("--") + 1 :] == ["python3", "-"]
    assert stdin == code.encode("utf-8")
    assert timeout == CONFIG.timeout


async def test_code_runs_as_an_unprivileged_user_in_tmp() -> None:
    runner = FakeRunner()

    await run_code_tool(CONFIG, runner=runner).run({"code": "print(1)"}, CTX)

    (exec_argv,) = runner.argv("exec")
    options = exec_argv[3 : exec_argv.index("--")]
    assert _option_values(options, "--user") == ["65534"]
    assert _option_values(options, "--group") == ["65534"]
    assert _option_values(options, "--cwd") == ["/tmp"]
    assert _option_values(options, "--env") == ["HOME=/tmp"]


async def test_sh_language_uses_the_shell() -> None:
    runner = FakeRunner()

    await run_code_tool(CONFIG, runner=runner).run({"code": "echo hi", "language": "sh"}, CTX)

    assert runner.argv("exec")[0][-2:] == ["sh", "-s"]


async def test_reports_stdout_stderr_and_exit_code() -> None:
    runner = FakeRunner(exec_result=CommandResult(3, "out line\n", "warn line\n"))

    result = await run_code_tool(CONFIG, runner=runner).run({"code": "x"}, CTX)

    assert result.content == "stdout:\nout line\n\nstderr:\nwarn line\n\nexit code: 3"


async def test_silent_script_reports_only_the_exit_code() -> None:
    result = await run_code_tool(CONFIG, runner=FakeRunner()).run({"code": "pass"}, CTX)

    assert result.content == "exit code: 0"


async def test_container_is_deleted_after_a_successful_run() -> None:
    runner = FakeRunner()

    await run_code_tool(CONFIG, runner=runner).run({"code": "print(1)"}, CTX)

    name = runner.argv("launch")[0][3]
    assert [argv[1] for argv, _, _ in runner.calls] == ["launch", "exec", "delete"]
    assert runner.calls[-1][0] == ["incus", "delete", "--force", name]


async def test_timeout_is_reported_and_container_still_deleted() -> None:
    runner = FakeRunner(exec_error=TimeoutError())

    result = await run_code_tool(CONFIG, runner=runner).run({"code": "while True: pass"}, CTX)

    assert result.content.startswith("Timed out after 12 seconds")
    name = runner.argv("launch")[0][3]
    assert runner.argv("delete") == [["incus", "delete", "--force", name]]


async def test_crash_during_exec_still_deletes_the_container() -> None:
    runner = FakeRunner(exec_error=OSError("incus vanished"))

    with pytest.raises(OSError):
        await run_code_tool(CONFIG, runner=runner).run({"code": "print(1)"}, CTX)

    assert len(runner.argv("delete")) == 1


async def test_launch_failure_raises_tool_error_without_running_code() -> None:
    runner = FakeRunner(launch=CommandResult(1, "", "Error: profile not found: /secret/path"))

    with pytest.raises(ToolError) as excinfo:
        await run_code_tool(CONFIG, runner=runner).run({"code": "print(1)"}, CTX)

    # Host details from stderr must not reach the model.
    assert "secret" not in str(excinfo.value)
    assert runner.argv("exec") == []
    assert len(runner.argv("delete")) == 1


async def test_launch_timeout_raises_tool_error_and_cleans_up() -> None:
    runner = FakeRunner(launch_error=TimeoutError())

    with pytest.raises(ToolError, match="too long to start"):
        await run_code_tool(CONFIG, runner=runner).run({"code": "print(1)"}, CTX)

    assert runner.argv("exec") == []
    assert len(runner.argv("delete")) == 1


async def test_cleanup_failure_does_not_hide_the_result() -> None:
    runner = FakeRunner(
        exec_result=CommandResult(0, "42\n", ""), delete_error=OSError("incus is gone")
    )

    result = await run_code_tool(CONFIG, runner=runner).run({"code": "print(42)"}, CTX)

    assert "42" in result.content


@pytest.mark.parametrize("language", ["ruby", "Python", "", None, 3, ["python"]])
async def test_unsupported_language_is_rejected_before_any_command(language: object) -> None:
    runner = FakeRunner()

    with pytest.raises(ToolError, match="language"):
        await run_code_tool(CONFIG, runner=runner).run({"code": "x", "language": language}, CTX)

    assert runner.calls == []


@pytest.mark.parametrize("args", [{}, {"code": ""}, {"code": "   \n"}, {"code": 42}])
async def test_missing_code_is_rejected_before_any_command(args: dict[str, object]) -> None:
    runner = FakeRunner()

    with pytest.raises(ToolError, match="code"):
        await run_code_tool(CONFIG, runner=runner).run(args, CTX)

    assert runner.calls == []


async def test_long_output_is_truncated() -> None:
    runner = FakeRunner(exec_result=CommandResult(1, "#" * 50_000, "!" * 50_000))

    result = await run_code_tool(CONFIG, runner=runner).run({"code": "x"}, CTX)

    assert result.content.count("[truncated]") == 2
    assert result.content.count("#") == MAX_OUTPUT_CHARS
    assert 0 < result.content.count("!") < MAX_OUTPUT_CHARS  # stderr gets the smaller share
    assert result.content.endswith("exit code: 1")


async def test_output_at_the_limit_is_not_truncated() -> None:
    runner = FakeRunner(exec_result=CommandResult(0, "#" * MAX_OUTPUT_CHARS, ""))

    result = await run_code_tool(CONFIG, runner=runner).run({"code": "x"}, CTX)

    assert "[truncated]" not in result.content


async def test_each_call_gets_its_own_container() -> None:
    runner = FakeRunner()
    tool = run_code_tool(CONFIG, runner=runner)

    await tool.run({"code": "print(1)"}, CTX)
    await tool.run({"code": "print(2)"}, CTX)

    first, second = (argv[3] for argv in runner.argv("launch"))
    assert first != second


# run_command is exercised against the local Python interpreter: no container,
# no network.


async def test_run_command_captures_output_and_exit_code() -> None:
    script = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"

    result = await run_command([sys.executable, "-c", script], None, 30.0)

    assert result.returncode == 3
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


async def test_run_command_feeds_stdin() -> None:
    result = await run_command([sys.executable, "-"], b"print(6 * 7)", 30.0)

    assert result.returncode == 0
    assert result.stdout.strip() == "42"


async def test_run_command_kills_the_process_on_timeout() -> None:
    with pytest.raises(TimeoutError):
        await run_command([sys.executable, "-c", "import time; time.sleep(60)"], None, 0.5)


async def test_run_command_caps_output_and_kills_a_flooding_process() -> None:
    flood = "import sys\nwhile True:\n    sys.stdout.write('x' * 1000)"

    # A process that never stops printing must end well before the timeout.
    result = await run_command([sys.executable, "-c", flood], None, 30.0, max_capture=10_000)

    assert result.stdout == "x" * 10_000


async def test_run_command_caps_each_stream_separately() -> None:
    script = "import sys; sys.stdout.write('o' * 5000); sys.stderr.write('e' * 5000)"

    result = await run_command([sys.executable, "-c", script], None, 30.0, max_capture=1000)

    assert result.stdout == "o" * 1000
    assert result.stderr == "e" * 1000

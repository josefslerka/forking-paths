#!/usr/bin/env python3
"""Run the deterministic v0.1 HTTP latency, payload, and RSS release gate."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GARDEN_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = GARDEN_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from apertus.garden.core.performance import (  # noqa: E402
    BUDGET_VERSION,
    MIB,
    STUB_PROCESS_BUDGET,
    STUB_SCENARIO_BUDGETS,
    evaluate_scenario,
    summarize_scenario,
)


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def process_rss_bytes(process_id: int) -> int:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(process_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = completed.stdout.strip()
    if not text:
        raise RuntimeError("Proces benchmarkového serveru už neběží.")
    return int(text.splitlines()[-1].strip()) * 1024


class HttpProbe:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> tuple[dict[str, Any], float, int]:
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Origin"] = self.base_url
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                response_body = response.read()
                status = response.status
        except HTTPError as exc:
            response_body = exc.read()
            raise RuntimeError(
                f"{path} vrátil HTTP {exc.code}: "
                f"{response_body.decode('utf-8', errors='replace')}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if status != 200:
            raise RuntimeError(f"{path} vrátil neočekávané HTTP {status}.")
        decoded = json.loads(response_body)
        if not isinstance(decoded, dict):
            raise RuntimeError(f"{path} nevrátil JSON objekt.")
        return decoded, elapsed_ms, len(response_body)


def wait_until_ready(
    probe: HttpProbe,
    process: subprocess.Popen[bytes],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Benchmarkový server skončil s kódem {process.returncode}."
            )
        try:
            health, _, _ = probe.request("/api/health", timeout=1.0)
            if health.get("status") == "ready":
                return health
        except (OSError, URLError, RuntimeError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(
        "Benchmarkový server nebyl připraven včas."
        + (f" Poslední chyba: {last_error}" if last_error else "")
    )


def identity_payload(health: dict[str, Any]) -> dict[str, Any]:
    active_model = health.get("active_model")
    if not isinstance(active_model, dict):
        raise RuntimeError("Health neobsahuje aktivní model.")
    return {
        "model_id": active_model["id"],
        "model_generation": health["model_generation"],
        "server_instance_id": health["server_instance_id"],
    }


def collect_scenario(
    operation: Callable[[int], tuple[dict[str, Any], float, int]],
    samples: int,
    rss_observations: list[int],
    process_id: int,
) -> dict[str, float | int]:
    latencies: list[float] = []
    sizes: list[int] = []
    for index in range(samples):
        _, elapsed_ms, response_size = operation(index)
        latencies.append(elapsed_ms)
        sizes.append(response_size)
        rss_observations.append(process_rss_bytes(process_id))
    return summarize_scenario(latencies, sizes)


def run_stub_budget(
    python_bin: str,
    *,
    json_output: Path | None,
) -> dict[str, Any]:
    port = free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="garden-performance-") as temporary:
        temporary_path = Path(temporary)
        database = temporary_path / "performance.sqlite3"
        log_path = temporary_path / "server.log"
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                [
                    python_bin,
                    str(GARDEN_DIR / "server.py"),
                    "--mock",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--database",
                    str(database),
                ],
                cwd=GARDEN_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                probe = HttpProbe(base_url)
                health = wait_until_ready(probe, process)
                identity = identity_payload(health)
                rss_observations = [process_rss_bytes(process.pid)]
                baseline_rss = rss_observations[-1]

                fixed_next = {
                    "text": "Výkonový rozpočet zahrady začíná",
                    "temperature": 1.0,
                    "top_k": 5,
                    **identity,
                }
                probe.request("/api/next", fixed_next)

                operations: dict[
                    str,
                    tuple[
                        int,
                        Callable[
                            [int],
                            tuple[dict[str, Any], float, int],
                        ],
                    ],
                ] = {
                    "health": (
                        9,
                        lambda _index: probe.request("/api/health"),
                    ),
                    "tokenization": (
                        7,
                        lambda index: probe.request(
                            "/api/tokenization",
                            {
                                "text_a": (
                                    f"Mám rád Prahu. Měření {index}."
                                ),
                                "text_b": (
                                    f"Ma\u0301m rád Prahu. Měření {index}."
                                ),
                            },
                        ),
                    ),
                    "next_token_uncached": (
                        7,
                        lambda index: probe.request(
                            "/api/next",
                            {
                                "text": (
                                    "Výkonový rozpočet — studený prefix "
                                    f"{index}"
                                ),
                                "temperature": 1.0,
                                "top_k": 5,
                                **identity,
                            },
                        ),
                    ),
                    "next_token_cached": (
                        9,
                        lambda _index: probe.request(
                            "/api/next",
                            fixed_next,
                        ),
                    ),
                    "text_surprisal": (
                        5,
                        lambda index: probe.request(
                            "/api/text-surprisal",
                            {
                                "text": (
                                    "Krátký teacher-forced text pro "
                                    f"výkonové měření {index}."
                                ),
                                "top_k": 5,
                                **identity,
                            },
                        ),
                    ),
                    "context_contrast": (
                        5,
                        lambda index: probe.request(
                            "/api/context-contrast",
                            {
                                "prefix_a": (
                                    "Hlavním městem České republiky je"
                                ),
                                "prefix_b": "Českou metropolí je",
                                "continuation": (
                                    f" Praha leží na Vltavě. {index}"
                                ),
                                "top_k": 5,
                                **identity,
                            },
                        ),
                    ),
                }

                scenario_results: dict[str, Any] = {}
                passed = True
                for name, (samples, operation) in operations.items():
                    summary = collect_scenario(
                        operation,
                        samples,
                        rss_observations,
                        process.pid,
                    )
                    evaluation = evaluate_scenario(
                        summary,
                        STUB_SCENARIO_BUDGETS[name],
                    )
                    scenario_results[name] = {
                        **summary,
                        **evaluation,
                    }
                    passed = passed and bool(evaluation["passed"])

                session_started = time.perf_counter()
                for index in range(
                    STUB_PROCESS_BUDGET.bounded_session_requests
                ):
                    probe.request(
                        "/api/next",
                        {
                            "text": (
                                "Dlouhá cache relace výkonového gate "
                                f"{index:03d}"
                            ),
                            "temperature": 1.0,
                            "top_k": 5,
                            **identity,
                        },
                    )
                    rss_observations.append(
                        process_rss_bytes(process.pid)
                    )
                session_ms = (
                    time.perf_counter() - session_started
                ) * 1_000
                peak_rss = max(rss_observations)
                rss_growth = max(0, peak_rss - baseline_rss)
                memory_violations: list[str] = []
                if peak_rss > STUB_PROCESS_BUDGET.max_rss_bytes:
                    memory_violations.append(
                        "peak_rss_bytes: "
                        f"{peak_rss} > "
                        f"{STUB_PROCESS_BUDGET.max_rss_bytes}"
                    )
                if (
                    rss_growth
                    > STUB_PROCESS_BUDGET.max_rss_growth_bytes
                ):
                    memory_violations.append(
                        "rss_growth_bytes: "
                        f"{rss_growth} > "
                        f"{STUB_PROCESS_BUDGET.max_rss_growth_bytes}"
                    )
                if (
                    session_ms
                    > STUB_PROCESS_BUDGET.bounded_session_ms
                ):
                    memory_violations.append(
                        "bounded_session_ms: "
                        f"{session_ms:.3f} > "
                        f"{STUB_PROCESS_BUDGET.bounded_session_ms:.3f}"
                    )
                memory_passed = not memory_violations
                passed = passed and memory_passed
                result = {
                    "profile": "stub",
                    "budget_version": BUDGET_VERSION,
                    "passed": passed,
                    "server": {
                        "mock": health.get("mock"),
                        "model_id": identity["model_id"],
                        "model_generation": identity[
                            "model_generation"
                        ],
                    },
                    "scenarios": scenario_results,
                    "process": {
                        "passed": memory_passed,
                        "violations": memory_violations,
                        "baseline_rss_bytes": baseline_rss,
                        "peak_rss_bytes": peak_rss,
                        "rss_growth_bytes": rss_growth,
                        "bounded_session_requests": (
                            STUB_PROCESS_BUDGET.bounded_session_requests
                        ),
                        "bounded_session_ms": session_ms,
                        "budget": {
                            "max_rss_bytes": (
                                STUB_PROCESS_BUDGET.max_rss_bytes
                            ),
                            "max_rss_growth_bytes": (
                                STUB_PROCESS_BUDGET.max_rss_growth_bytes
                            ),
                            "bounded_session_ms": (
                                STUB_PROCESS_BUDGET.bounded_session_ms
                            ),
                        },
                    },
                }
            except Exception:
                process.terminate()
                process.wait(timeout=10)
                log_file.flush()
                try:
                    server_log = log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError:
                    server_log = ""
                if server_log:
                    print(server_log, file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def print_report(result: dict[str, Any]) -> None:
    state = "PASS" if result["passed"] else "FAIL"
    print(
        f"Performance gate {result['budget_version']} "
        f"({result['profile']}): {state}"
    )
    for name, scenario in result["scenarios"].items():
        marker = "✓" if scenario["passed"] else "✗"
        print(
            f"  {marker} {name:22} "
            f"median {scenario['median_ms']:7.2f} ms · "
            f"p95 {scenario['p95_ms']:7.2f} ms · "
            f"max payload {scenario['max_response_bytes']:8d} B"
        )
        for violation in scenario["violations"]:
            print(f"      {violation}")
    process = result["process"]
    marker = "✓" if process["passed"] else "✗"
    print(
        f"  {marker} bounded cache session    "
        f"{process['bounded_session_ms']:.2f} ms · "
        f"peak {process['peak_rss_bytes'] / MIB:.1f} MiB · "
        f"growth {process['rss_growth_bytes'] / MIB:.1f} MiB"
    )
    for violation in process["violations"]:
        print(f"      {violation}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=os.environ.get("GARDEN_PYTHON", sys.executable),
        help="Python použitý ke spuštění izolovaného mock serveru.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Volitelná cesta pro strojově čitelný report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stub_budget(
        args.python,
        json_output=args.json_output,
    )
    print_report(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

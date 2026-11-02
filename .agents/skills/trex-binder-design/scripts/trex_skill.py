#!/usr/bin/env python3
"""Standalone binder-design workflow: bundled control, direct backends, Codex decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from trex_codex.mailbox import DEFAULT_DECISION_TIMEOUT_SECONDS, MAX_DECISION_TIMEOUT_SECONDS


SHOW_PAGE_DEFAULT_LIMIT = 6000
SHOW_PAGE_MAX_LIMIT = 8000


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Prepare a campaign with Codex as planner and supervisor")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--target", required=True)
    init.add_argument("--target-constraint", type=Path)
    init.add_argument("--target-pdb", type=Path)
    init.add_argument("--worker-gpus", required=True, help="Explicit allocation-visible IDs, e.g. 0,1 or GPU UUIDs")
    init.add_argument("--hours", type=float, required=True)
    init.add_argument("--output", type=Path, required=True, help="Exact archive directory")
    init.add_argument("--name")
    init.add_argument("--asset-root", type=Path)
    init.add_argument("--families", help="Comma-separated supported action families")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--agent-model", help="Known Codex model label; omit if unknown")

    check = commands.add_parser("check", help="Strict backend/input checks for the bundled workflow")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--agent-model")

    run = commands.add_parser("start", aliases=["run"], help="Run inside an existing GPU allocation; Codex must service requests")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--mailbox", type=Path, required=True)
    run.add_argument("--decision-timeout", type=float, default=DEFAULT_DECISION_TIMEOUT_SECONDS,
                     help=f"Seconds per decision, in (0, {MAX_DECISION_TIMEOUT_SECONDS:g}]; "
                          f"default {DEFAULT_DECISION_TIMEOUT_SECONDS:g}; bounded by campaign time")
    run.add_argument("--agent-model")
    run.add_argument("--maximum-iterations", type=int, help="Optional additional controller-loop bound")
    run.add_argument("--evaluation-only", action="store_true",
                     help="Finish existing canonical AF2 evaluations without new generation; persists on resume")

    stop = commands.add_parser("stop", help="Stop new campaign work and drain running workers")
    stop.add_argument("--archive", type=Path, required=True)

    drain = commands.add_parser("drain", help="Stop generation in an active campaign and finish existing canonical evaluations")
    drain.add_argument("--archive", type=Path, required=True)

    status = commands.add_parser("status", help="Read archive and execution counts")
    status.add_argument("--archive", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="Inspect/validate an archive or trace decisions")
    analyze.add_argument("action", choices=["summary", "validate", "trace", "schema"])
    analyze.add_argument("--archive", type=Path)
    analyze.add_argument("--limit", type=int, default=10)
    export = commands.add_parser("export", help="Export ranked candidates with the bundled exporter")
    export.add_argument("export_args", nargs=argparse.REMAINDER,
                        help="Exporter arguments after --; use export -- --help")

    requests = commands.add_parser("requests", help="List pending decisions (or history with --all)")
    requests.add_argument("--mailbox", type=Path, required=True)
    requests.add_argument("--all", action="store_true")
    for name, help_text in (
        ("show", "Read the authoritative decision request, optionally in pages"),
        ("respond", "Submit a Codex-authored JSON decision exactly once"),
        ("cancel", "Cancel a waiting decision and stop new controller work"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--mailbox", type=Path, required=True)
        sub.add_argument("--request-id", required=True)
        if name == "show":
            sub.add_argument("--offset", type=int,
                             help="Enable paging at this Unicode codepoint offset (default 0)")
            sub.add_argument("--limit", type=int,
                             help=f"Enable paging with 1–{SHOW_PAGE_MAX_LIMIT} codepoints per page "
                                  f"(default {SHOW_PAGE_DEFAULT_LIMIT})")
        elif name == "respond":
            sub.add_argument("--response", required=True, help="JSON file path, or - to read stdin")
    return root


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _request_page(request: dict[str, object], offset: int, limit: int) -> dict[str, object]:
    """Slice verified immutable request content; lifecycle updates stay outside its hash."""
    if offset < 0:
        raise ValueError("--offset must be nonnegative")
    if not 1 <= limit <= SHOW_PAGE_MAX_LIMIT:
        raise ValueError(f"--limit must be between 1 and {SHOW_PAGE_MAX_LIMIT}")
    payload = {key: value for key, value in request.items()
               if key not in {"status", "status_updated_at", "reason"}}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    total = len(text)
    if offset > total:
        raise ValueError(f"--offset exceeds request length ({total} codepoints)")
    end = min(offset + limit, total)
    return {
        **{key: request[key] for key in (
            "version", "request_id", "session_id", "role", "prompt_hash",
            "status", "deadline_at", "deadline_unix",
        )},
        "payload_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "total": total,
        "offset": offset,
        "next_offset": end if end < total else None,
        "text": text[offset:end],
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in {"requests", "show", "respond", "cancel"}:
            from trex_codex.mailbox import cancel_request, list_requests, read_request, submit_response

            if args.command == "requests":
                entries = list_requests(args.mailbox)
                _json(entries if args.all else [item for item in entries if item["status"] in {"waiting", "invalid"}])
            elif args.command == "show":
                request = read_request(args.mailbox, args.request_id)
                if args.offset is None and args.limit is None:
                    _json(request)
                else:
                    _json(_request_page(
                        request, args.offset if args.offset is not None else 0,
                        args.limit if args.limit is not None else SHOW_PAGE_DEFAULT_LIMIT,
                    ))
            elif args.command == "cancel":
                _json(cancel_request(args.mailbox, args.request_id))
            else:
                # Bound input before parsing, including for stdin on an SSH connection.
                if args.response == "-":
                    raw = sys.stdin.read(1_048_577)
                else:
                    with Path(args.response).open(encoding="utf-8") as source:
                        raw = source.read(1_048_577)
                if len(raw.encode("utf-8")) > 1_048_576:
                    raise ValueError("Response exceeds 1 MiB")
                # Windows PowerShell 5 writes a BOM with Set-Content -Encoding UTF8.
                # Accept it for both transferred files and SSH stdin; keep the byte limit above.
                payload = json.loads(raw.removeprefix("\ufeff"))
                if not isinstance(payload, dict):
                    raise ValueError("Response must be a JSON object")
                _json(submit_response(args.mailbox, args.request_id, payload))
            return 0

        if args.command == "status":
            from binder_runtime.campaign.status import collect_campaign_status
            _json(collect_campaign_status(args.archive).as_dict())
            return 0
        if args.command == "analyze":
            from binder_runtime.analysis import main as analyze_main
            arguments = [args.action, "--json"]
            if args.action != "schema":
                if args.archive is None:
                    raise ValueError("--archive is required for this analysis")
                if not args.archive.is_dir():
                    raise ValueError(f"Archive does not exist: {args.archive}")
                arguments += ["--archive-root", str(args.archive)]
            if args.action == "trace":
                arguments += ["--limit", str(args.limit)]
            return analyze_main(arguments)
        if args.command == "export":
            from binder_runtime.export_best_n import main as export_main
            arguments = args.export_args
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            return export_main(arguments)
        from trex_codex import runner

        if args.command == "stop":
            _json(runner.stop_campaign(args.archive))
        elif args.command == "drain":
            _json(runner.drain_campaign(args.archive))
        elif args.command == "init":
            out = runner.init_campaign(
                args.config, target=args.target, archive_root=args.output,
                worker_gpus=args.worker_gpus, hours=args.hours, name=args.name,
                asset_root=args.asset_root, target_constraint=args.target_constraint,
                target_pdb=args.target_pdb,
                families=args.families.split(",") if args.families else None,
                seed=args.seed, model=args.agent_model,
            )
            _json({"config": str(out), "decision_provider": "current Codex conversation"})
        elif args.command == "check":
            report = runner.check_campaign(args.config, strict=True, model=args.agent_model)
            _json(report)
            return 0 if report["ok"] else 2
        elif args.command in {"start", "run"}:
            return runner.run_campaign(
                args.config, mailbox_root=args.mailbox,
                timeout_seconds=args.decision_timeout, model=args.agent_model,
                maximum_iterations=args.maximum_iterations,
                evaluation_only=args.evaluation_only,
            )
        return 0
    except KeyboardInterrupt:
        print("trex-skill: interrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError, RecursionError, subprocess.SubprocessError) as exc:
        print(f"trex-skill: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import ConfigurationError, Settings
from .emailer import EmailSender
from .http_client import HttpClient
from .llm import OpenRouterClassifier
from .pipeline import PipelineError, PipelineReport, TrackerPipeline
from .sources import ArxivSource, HuggingFaceSource
from .storage import StateStore
from .time_utils import parse_datetime, utc_now

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-tracker",
        description="Track useful diffusion-model optimization papers and email them daily.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the tracker")
    run_parser.add_argument("--send", action="store_true", help="Send the report by SMTP")
    run_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use the deterministic heuristic without calling OpenRouter",
    )
    run_parser.add_argument(
        "--source",
        action="append",
        choices=("arxiv", "huggingface"),
        help="Limit collection to one source; may be repeated",
    )
    run_parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    run_parser.add_argument(
        "--now",
        help="Override the run time with an ISO-8601 timestamp, mainly for testing",
    )
    run_parser.add_argument("--state-db", help="Override STATE_DB_PATH")
    run_parser.add_argument(
        "--lookback-hours",
        type=int,
        help="Override LOOKBACK_HOURS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command != "run":
        parser.error("A command is required")
    try:
        settings = Settings.from_env()
        overrides: dict[str, object] = {}
        if args.no_llm:
            overrides["llm_enabled"] = False
        if args.state_db:
            overrides["state_db_path"] = args.state_db
        if args.lookback_hours is not None:
            if args.lookback_hours <= 0:
                raise ConfigurationError("--lookback-hours must be positive")
            overrides["lookback_hours"] = args.lookback_hours
        settings = settings.with_overrides(**overrides)
        now = parse_datetime(args.now) if args.now else utc_now()
        if now is None:
            raise ConfigurationError("--now must be a valid ISO-8601 timestamp")
        source_names = tuple(args.source or ("arxiv", "huggingface"))
        http_client = HttpClient(
            user_agent=settings.user_agent,
            retries=max(3, settings.llm_max_retries),
        )
        classifier = OpenRouterClassifier(settings, http_client) if settings.llm_enabled else None
        arxiv_source = ArxivSource(settings, http_client)
        huggingface_source = HuggingFaceSource(settings, http_client)
        store = StateStore(settings.state_db_path)
        pipeline = TrackerPipeline(
            settings=settings,
            arxiv_source=arxiv_source,
            huggingface_source=huggingface_source,
            classifier=classifier,
            store=store,
            email_sender=EmailSender(settings),
        )
        try:
            report = pipeline.run(send=args.send, now=now, source_names=source_names)
        finally:
            store.close()
            http_client.close()
        _print_report(report, args.json)
        return 0
    except (ConfigurationError, PipelineError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1


def _print_report(report: PipelineReport, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=True, indent=2))
        return
    data = report.to_dict()
    print(
        f"Fetched {data['fetched_count']} papers; "
        f"{data['candidate_count']} candidates; {data['useful_count']} useful; "
        f"{data['sent_count']} sent."
    )
    for paper in data["selected"]:
        print(f"- {paper['title']} [{paper['useful_category']}]")
    for error in data["errors"]:
        print(f"Warning: {error}")

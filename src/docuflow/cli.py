"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from docuflow.bootstrap import bootstrap
from docuflow.config import get_settings
from docuflow.ingest.fetch import FetchError, fetch_url
from docuflow.llm.factory import describe_llms
from docuflow.masterdata import MasterDataRepository
from docuflow.ocr.factory import describe_providers
from docuflow.pipeline import Pipeline, PipelineResult
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

DECISION_LABEL = {
    "auto_post": "POSTED",
    "needs_review": "REVIEW",
    "rejected": "REJECTED",
}


def _print_result(result: PipelineResult, verbose: bool = True) -> None:
    run = result.run
    print(f"\n{'=' * 78}")
    print(f"{run.source.filename}  ->  {DECISION_LABEL[run.decision]}  ({run.confidence:.0%} confidence)")
    print(f"{'=' * 78}")
    print(f"Type      : {run.doc_type} ({run.doc_type_confidence:.0%})")
    print(f"OCR       : {run.ocr.provider}, {run.ocr.char_count:,} chars, {run.ocr.duration_ms:,} ms")
    if result.vendor_match.matched:
        print(f"Vendor    : {result.vendor_match.vendor['legal_name']} (via {result.vendor_match.method})")
    if run.record_id:
        print(f"Record    : {run.record_id}")

    if verbose:
        print(f"\n{'FIELD':22} {'VALUE':32} {'CONF':>6}  {'SOURCE':<8} EVIDENCE")
        print("-" * 100)
        for field in run.fields:
            if field.is_empty:
                continue
            mark = " " if field.grounded else "!"
            evidence = (field.evidence or "")[:28].replace("\n", " ")
            print(
                f"{mark}{field.name:21} {str(field.value)[:31]:32} {field.confidence:6.0%}  "
                f"{field.source:<8} {evidence}"
            )

    failed = run.failed_findings
    if failed:
        print(f"\nCHECKS ({len(run.findings) - len(failed)}/{len(run.findings)} passed)")
        for finding in failed:
            print(f"  [{finding.severity.upper():8}] {finding.rule_id}: {finding.message}")

    if result.reconciliation.assessment:
        print(f"\nANALYST: {result.reconciliation.assessment}")
        if result.reconciliation.reviewer_note:
            print(f"ACTION : {result.reconciliation.reviewer_note}")
        for correction in result.reconciliation.corrections:
            flag = "verified" if correction.grounded else "UNVERIFIED"
            print(f"  suggest {correction.field} = {correction.suggested_value!r} [{flag}] - {correction.reason}")

    print()


def cmd_init(args: argparse.Namespace) -> int:
    settings = get_settings()
    result = bootstrap(settings, with_samples=not args.no_samples)
    print(json.dumps(result, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"Database          : {settings.db_path}")
    print(f"Auto-post at      : {settings.auto_post_min_confidence:.0%} confidence")
    print(f"Amount tolerance  : {settings.amount_tolerance:.0%}\n")

    print("OCR providers:")
    for row in describe_providers(settings):
        mark = "*" if row["active"] else " "
        state = "ready" if row["available"] else f"unavailable - {row['reason']}"
        print(f"  {mark} {row['key']:12} {state}")

    print("\nLLM providers:")
    for row in describe_llms(settings):
        mark = "*" if row["active"] else " "
        state = "ready" if row["available"] else f"unavailable - {row['reason']}"
        print(f"  {mark} {row['key']:12} {state}")

    if settings.db_path.exists():
        stats = DocumentStore(settings.db_path).stats()
        print(f"\nProcessed {stats['total_documents']} documents; "
              f"auto-post rate {stats['auto_post_rate']:.0%}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    settings = get_settings()
    bootstrap(settings, with_samples=False)
    pipeline = Pipeline(settings)

    def progress(step: str, message: str) -> None:
        if not args.quiet:
            print(f"  ... {message}", file=sys.stderr)

    target = args.target
    if target.startswith(("http://", "https://")):
        try:
            fetched = fetch_url(target, settings.max_document_bytes)
        except FetchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        result = pipeline.process_bytes(
            fetched.content,
            fetched.filename,
            source_kind="url",
            origin=fetched.final_url,
            content_type=fetched.content_type,
            on_progress=progress,
        )
    else:
        path = Path(target)
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        result = pipeline.process_file(path, on_progress=progress)

    if args.json:
        print(json.dumps(result.run.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_result(result, verbose=not args.quiet)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    settings = get_settings()
    bootstrap(settings, with_samples=True)

    if args.reset:
        from docuflow.db import transaction

        with transaction(settings.db_path) as connection:
            for table in ("audit_log", "corrections", "records", "documents"):
                connection.execute(f"DELETE FROM {table}")
        print("Cleared previously processed documents and posted records.\n")

    pipeline = Pipeline(settings)

    manifest_paths = sorted(settings.sample_dir.glob("*.pdf")) + sorted(settings.sample_dir.glob("*.jpg"))
    # Prefer the scanned artefact when both exist for one sample.
    by_stem: dict[str, Path] = {}
    for path in manifest_paths:
        if path.suffix == ".jpg" or path.stem not in by_stem:
            by_stem[path.stem] = path

    targets = sorted(by_stem.values())
    if args.limit:
        targets = targets[: args.limit]

    print(f"Processing {len(targets)} sample documents...\n")
    summary: list[tuple[str, str, float]] = []
    for path in targets:
        print(f"--> {path.name}", file=sys.stderr)
        result = pipeline.process_file(path, source_kind="sample")
        _print_result(result, verbose=not args.quiet)
        summary.append((path.name, result.run.decision, result.run.confidence))

    print("=" * 78)
    print(f"{'DOCUMENT':40} {'DECISION':14} CONFIDENCE")
    print("-" * 78)
    for name, decision, confidence in summary:
        print(f"{name:40} {DECISION_LABEL[decision]:14} {confidence:.0%}")
    posted = sum(1 for _, decision, _ in summary if decision == "auto_post")
    print(f"\nStraight-through: {posted}/{len(summary)} ({posted / max(len(summary), 1):.0%})")
    return 0


def cmd_records(args: argparse.Namespace) -> int:
    settings = get_settings()
    records = RecordStore(settings.db_path).list()
    if not records:
        print("No records posted yet.")
        return 0
    print(f"{'RECORD':16} {'KIND':12} {'REFERENCE':20} {'COUNTERPARTY':32} {'AMOUNT':>14}")
    print("-" * 100)
    for record in records:
        amount = f"{record['amount']:,.2f} {record['currency'] or ''}" if record["amount"] else "-"
        print(
            f"{record['record_id']:16} {record['record_kind']:12} "
            f"{str(record['reference'] or '-')[:19]:20} {str(record['counterparty'] or '-')[:31]:32} {amount:>14}"
        )
    return 0


def _port_is_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _describe_port_holder(port: int) -> str:
    """Best-effort description of whatever already owns the port."""
    import shutil
    import subprocess

    if shutil.which("lsof") is None:
        return ""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    lines = [line for line in result.stdout.splitlines()[1:] if line.strip()]
    if not lines:
        return ""
    fields = lines[0].split()
    return f" (held by {fields[0]}, PID {fields[1]})" if len(fields) > 1 else ""


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()

    # Bind failures surface from deep inside uvicorn as a bare traceback, which
    # is a poor first experience for something as ordinary as a busy port.
    if not _port_is_free(args.host, args.port):
        holder = _describe_port_holder(args.port)
        print(
            f"error: port {args.port} on {args.host} is already in use{holder}.\n"
            f"       Start on another port:  docuflow serve --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    bootstrap(settings, with_samples=not args.no_samples)
    print(f"DocuFlow running at http://{args.host}:{args.port}")
    uvicorn.run("docuflow.api:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from docuflow.evaluation.harness import run_evaluation
    from docuflow.evaluation.report import print_report

    settings = get_settings()
    bootstrap(settings, with_samples=True)
    report = run_evaluation(settings, limit=args.limit)
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


def cmd_vendor(args: argparse.Namespace) -> int:
    """Manage the vendor master file, so unknown counterparties can be registered."""
    settings = get_settings()
    bootstrap(settings, with_samples=False)
    master = MasterDataRepository(settings.db_path)

    if args.vendor_command == "list":
        vendors = master.list_vendors()
        print(f"{'ID':10} {'LEGAL NAME':44} {'TAX ID':16} {'ACCOUNT':24} STATUS")
        print("-" * 108)
        for vendor in vendors:
            print(
                f"{vendor['vendor_id']:10} {vendor['legal_name'][:43]:44} "
                f"{str(vendor['tax_id'] or '-'):16} {str(vendor['bank_account'] or '-'):24} {vendor['status']}"
            )
        print(f"\n{len(vendors)} vendor(s).")
        return 0

    try:
        vendor = master.add_vendor(
            legal_name=args.name,
            tax_id=args.tax_id,
            bank_account=args.account,
            aliases=args.alias or "",
            status=args.status,
            payment_terms=args.terms,
            currency=args.currency,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Registered {vendor['legal_name']} as {vendor['vendor_id']}.")
    print("Documents from this vendor will now match instead of being blocked.")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from docuflow.models_registry import list_models

    settings = get_settings()
    catalog = list_models(settings)
    for kind, title in (("ocr", "OCR ENGINES"), ("llm", "AGENT MODELS")):
        print(f"\n{title}")
        print("-" * 78)
        for option in catalog[kind]:
            mark = "*" if option["recommended"] else " "
            state = "ready" if option["available"] else f"unavailable ({option['reason']})"
            tags = f"  [{', '.join(option['tags'])}]" if option["tags"] else ""
            print(f" {mark} {option['key']:52} {state}{tags}")
    print("\n* = recommended.  Select with OCR_PROVIDER / LLM_PROVIDER in .env, or the UI dropdown.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from docuflow.evaluation.benchmark import (
        Combination,
        parse_combination,
        print_benchmark,
        run_benchmark,
    )

    settings = get_settings()
    if args.combo:
        combinations = [parse_combination(spec) for spec in args.combo]
    else:
        # A sensible default sweep across whatever is installed.
        combinations = [
            Combination("typhoon cloud OCR", "typhoon_api", settings.ocr_cloud_model),
            Combination("ollama typhoon-ocr1.5-3b", "ollama", settings.ocr_ollama_model),
            Combination("mlx typhoon-ocr1.5-2b", "local", settings.ocr_local_model),
            Combination("pdf text layer", "pdftext"),
        ]

    result = run_benchmark(settings, combinations, limit=args.limit)
    print_benchmark(result)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docuflow", description="Document extraction pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create the database and demo corpus")
    init_parser.add_argument("--no-samples", action="store_true")
    init_parser.set_defaults(func=cmd_init)

    status_parser = subparsers.add_parser("status", help="show configured providers")
    status_parser.set_defaults(func=cmd_status)

    process_parser = subparsers.add_parser("process", help="process one file or URL")
    process_parser.add_argument("target", help="path to a document, or an http(s) URL")
    process_parser.add_argument("--json", action="store_true", help="emit the full run as JSON")
    process_parser.add_argument("--quiet", action="store_true")
    process_parser.set_defaults(func=cmd_process)

    demo_parser = subparsers.add_parser("demo", help="run the whole sample corpus")
    demo_parser.add_argument("--limit", type=int, default=0)
    demo_parser.add_argument("--quiet", action="store_true")
    demo_parser.add_argument(
        "--reset", action="store_true", help="clear processed documents and records first"
    )
    demo_parser.set_defaults(func=cmd_demo)

    records_parser = subparsers.add_parser("records", help="list posted records")
    records_parser.set_defaults(func=cmd_records)

    serve_parser = subparsers.add_parser("serve", help="start the web platform")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--no-samples", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    eval_parser = subparsers.add_parser("evaluate", help="measure accuracy against ground truth")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--out", help="write the JSON report here")
    eval_parser.set_defaults(func=cmd_evaluate)

    vendor_parser = subparsers.add_parser("vendor", help="manage the vendor master file")
    vendor_sub = vendor_parser.add_subparsers(dest="vendor_command", required=True)
    vendor_sub.add_parser("list", help="list registered vendors")
    vendor_add = vendor_sub.add_parser("add", help="register a vendor")
    vendor_add.add_argument("name", help="legal name as printed on documents")
    vendor_add.add_argument("--tax-id", help="tax identification number")
    vendor_add.add_argument("--account", help="registered bank account")
    vendor_add.add_argument("--alias", help="other names, separated by ';'")
    vendor_add.add_argument("--status", default="active", choices=["active", "on_hold", "blocked"])
    vendor_add.add_argument("--terms", help="payment terms, e.g. 'Net 30'")
    vendor_add.add_argument("--currency", default="THB")
    vendor_parser.set_defaults(func=cmd_vendor)

    models_parser = subparsers.add_parser("models", help="list selectable OCR and agent models")
    models_parser.set_defaults(func=cmd_models)

    bench_parser = subparsers.add_parser("benchmark", help="compare models on the ground-truth corpus")
    bench_parser.add_argument(
        "--combo",
        action="append",
        help='e.g. --combo "ocr=ollama:scb10x/typhoon-ocr1.5-3b,llm=typhoon" (repeatable)',
    )
    bench_parser.add_argument("--limit", type=int, default=0, help="only the first N documents")
    bench_parser.add_argument("--out", help="write the JSON comparison here")
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

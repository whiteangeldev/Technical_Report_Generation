from __future__ import annotations

import argparse
import sys

from trg.sections import SECTION_IDS, assemble_report
from trg.paths import OUTPUT_DIR, ensure_output_dirs


def _add_section_arg(parser: argparse.ArgumentParser, required: bool = False) -> None:
    parser.add_argument(
        "-s",
        "--section",
        choices=SECTION_IDS,
        help="Single section id (default: all sections where applicable)",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    from trg.generate import generate_section

    targets = [args.section] if args.section else SECTION_IDS
    for sid in targets:
        print(f"Generating {sid}...")
        generate_section(sid)
    print("Done.")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from trg.expert_correction import review_all, review_section

    if args.section:
        result = review_section(args.section, apply_fix=args.apply)
        print(f"Score: {result.get('score')}; saved {result.get('review_file')}")
        if args.apply:
            print(f"Applied correction to sections/{args.section}/section.md")
    else:
        for result in review_all(apply_fix=args.apply):
            sid = result.get("review_file", "").split("/")[-1].split("_")[0]
            print(f"{sid}: score={result.get('score')} -> {result.get('review_file')}")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    from trg.fact_alignment import align_facts, write_alignment_summary

    if args.section:
        results = [align_facts(args.section)]
    else:
        results = [align_facts()]
    summary = write_alignment_summary(results)
    aligned = results[0].get("aligned")
    print(f"Aligned: {aligned}; summary -> {summary}")
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    from trg.question_generator import generate_questions

    path = generate_questions(min_questions=args.min)
    print(f"Questions written -> {path}")
    return 0


def cmd_search_refs(args: argparse.Namespace) -> int:
    from trg.reference_search import run_search_refs

    path = run_search_refs()
    print(f"References saved -> {path}")
    print("Next: download PDFs from DOIs, convert to Markdown under sources/papers/")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    report = assemble_report()
    out = OUTPUT_DIR / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report assembled -> {out}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Generate all sections, review, align facts, questions, assemble report."""
    cmd_generate(argparse.Namespace(section=None))
    cmd_review(argparse.Namespace(section=None, apply=args.apply))
    cmd_align(argparse.Namespace(section=None))
    cmd_questions(argparse.Namespace(min=args.min_questions))
    cmd_report(argparse.Namespace())
    print("Pipeline complete. See output/ and sections/.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trg",
        description="Cursor-friendly automated technical report generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Draft section content with the LLM")
    _add_section_arg(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_rev = sub.add_parser("review", help="Expert-perspective review and optional fix")
    _add_section_arg(p_rev)
    p_rev.add_argument("--apply", action="store_true", help="Write corrected_markdown to section")
    p_rev.set_defaults(func=cmd_review)

    p_align = sub.add_parser("align-facts", help="Rule-based fact alignment check")
    _add_section_arg(p_align)
    p_align.set_defaults(func=cmd_align)

    p_q = sub.add_parser("questions", help="Generate a long expert question list")
    p_q.add_argument("--min", type=int, default=40, help="Minimum number of questions")
    p_q.set_defaults(func=cmd_questions)

    p_ref = sub.add_parser(
        "search-refs",
        help="Search Crossref and save title/topic/DOI to output/references.xlsx",
    )
    p_ref.set_defaults(func=cmd_search_refs)

    p_rep = sub.add_parser("report", help="Assemble sections into output/report.md")
    p_rep.set_defaults(func=cmd_report)

    p_all = sub.add_parser("pipeline", help="Run generate → review → align → questions → report")
    p_all.add_argument("--apply", action="store_true", help="Apply expert corrections to sections")
    p_all.add_argument("--min-questions", type=int, default=40)
    p_all.set_defaults(func=cmd_pipeline)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

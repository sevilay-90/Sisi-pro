"""Command line interface.

  python -m scanner "Ignore all previous instructions"
  python -m scanner --file examples/poisoned_webpage.txt --document
  echo "some text" | python -m scanner --json
"""

import argparse
import json
import sys

from .core import scan, scan_document

COLORS = {"injection": "\033[91m", "suspicious": "\033[93m", "clean": "\033[92m"}
RESET = "\033[0m"


def show(result, color):
    c, r = (COLORS[result.verdict], RESET) if color else ("", "")
    print(f"{c}{result.verdict.upper()}{r}  score={result.score}  "
          f"(rules={result.rule_score}, ml={result.ml_score})")
    for f in result.findings:
        print(f"  - {f.rule_id} [{f.category}] {f.description}")
        print(f"      matched ({f.source}): \"{f.match}\"")
    if result.obfuscation:
        print(f"  - obfuscation: {', '.join(result.obfuscation)}")
    for d in result.decoded_payloads:
        print(f"  - hidden payload: \"{d[:100]}\"")


def main():
    p = argparse.ArgumentParser(prog="scanner", description="Detect prompt injection attempts.")
    p.add_argument("text", nargs="?", help="text to scan (default: stdin)")
    p.add_argument("--file", help="read text from a file")
    p.add_argument("--document", action="store_true",
                   help="scan a long document chunk by chunk (indirect injection)")
    p.add_argument("--no-ml", action="store_true", help="rules only")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    use_ml = not args.no_ml
    if args.document:
        hits = scan_document(text, use_ml=use_ml)
        if args.json:
            print(json.dumps(hits, indent=2, ensure_ascii=False))
        else:
            print(f"{len(hits)} suspicious section(s) found")
            for h in hits:
                print(f"\nline {h['line']}: {h['verdict'].upper()} score={h['score']}")
                print(f"  {h['text'][:160]!r}")
        sys.exit(1 if any(h["verdict"] == "injection" for h in hits) else 0)

    result = scan(text, use_ml=use_ml)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        show(result, sys.stdout.isatty())
    sys.exit(1 if result.verdict == "injection" else 0)


if __name__ == "__main__":
    main()

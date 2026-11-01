"""Sanitized public export of the T5 form-cost experiment.

Usage:
    python tools/sanitize_export.py SOURCE_DIR TARGET_DIR \
        --redact "<producer machine name>=producer-node" \
        --redact "<private runtime folder name>=T5Agents"

The private literals are given on the command line, not stored here, so that this public
script does not republish what it removes.

SOURCE_DIR is the private working folder of the experiment; TARGET_DIR is the public
folder (this one). The script copies an explicit allow-list of files, rewrites only
provenance stamps, local absolute paths, internal run codes and private wiki links, and
leaves every number, seed, statistic and hash string untouched. JSON files are rewritten
at text level (formatting and line endings are preserved), then parsed again and compared
with an object-level transform of the original, so a sanitized JSON can differ from its
source only in the documented fields. The source is opened read-only.

It prints one line per file: relative path, original sha256, exported sha256.
Hand-written public files (README.md, SANITIZATION.md, tools/) are not touched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

PRODUCER = "producer-node"
SEP = r"(?:\\\\|\\|/)+"  # escaped JSON backslash, Markdown backslash or slash
REST = r"[^\s\"'`()<>|\]]*"

EXPLICIT = [
    "t5.py",
    "PREREGISTRATION.md",
    "PREREGISTRATION-v2.md",
    "DEV-AMENDMENT-2026-10-23.md",
    "ANALYSIS-ADDENDUM-2026-10-23.md",
    "RESULTS.md",
    "pilot-analysis.md",
    "final-summary.json",
    "pilot-summary.json",
    "runs/analyze.py",
    "runs/driver.py",
    "runs/test_analysis.py",
    "runs/confirmatory-manifest.json",
    "runs/pilot-manifest.json",
    "runs/pilot-v2-manifest.json",
    "runs/confirmatory-gate.json",
    "runs/confirmation-data-hashes.json",
    "runs/confirmation-independent.json",
]
RAW_GLOBS = [
    "runs/confirmatory/native-retention-v2/seed*_*.json",
    "runs/pilot-dev/native-retention-v2/seed*_*.json",
    "runs/pilot-dev/seed*_*.json",
]
MAX_BYTES = 5 * 1024 * 1024


SEG = r"[^\\/\s\"'`]+"
DRIVE = r"(?<![A-Za-z])[A-Za-z]:"
# Any absolute Windows path prefix, lazily, up to a separator: <drive>:/dir/dir/
PREFIX = DRIVE + r"(?:" + SEP + SEG + r")*?" + SEP
TAIL = r"(?:" + SEP + r"(" + REST + r"))?"
RUN = r"[Hh]\d{5}"  # internal session/run code

PATH_RULES = [
    # private runtime folder of the driver -> <local>/t5-form-cost
    (re.compile(PREFIX + r"t5-form-cost-" + RUN), "<local>/t5-form-cost"),
    (re.compile(r"%LOCALAPPDATA%(?:" + SEP + SEG + r")*?" + SEP + r"t5-form-cost-" + RUN), "<local>/t5-form-cost"),
    # experiment root -> path relative to this folder
    (re.compile(PREFIX + r"t5-form-cost(?![\w-])" + TAIL), "t5root"),
    # private scratch folder of the run -> <local>/<rest>
    (re.compile(PREFIX + r"t5-" + RUN + TAIL), "local-rest"),
    (re.compile(PREFIX + r"machine-pedagogy(?![\w-])"), "machine-pedagogy"),
    (re.compile(PREFIX + r"python\.exe"), "python"),
    # any other absolute local path -> <local>/<file name>
    (re.compile(DRIVE + SEP + r"(" + REST + r")"), "local-basename"),
    (re.compile(r"/Users/[^\s\"'`]+"), "<local>"),
]
RUNCODE_RULES = [
    (re.compile(r"T5-form-cost-" + RUN), "T5-form-cost"),
    (re.compile(r"t5-form-cost-" + RUN), "t5-form-cost"),
    (re.compile(r"t5-" + RUN), "t5"),
    (re.compile(r"(?<![0-9A-Za-z])" + RUN + r"(?![0-9A-Za-z])"), "this run"),
]
FRONTMATTER_DROP = {"machine", "operator", "authored_by", "origin"}
# Private literals (producer machine name, private folder names) are passed on the command
# line with --redact OLD=NEW so that this public script does not itself publish them.
REDACT = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(rest):
    return re.sub(SEP, "/", rest or "")


def fix_paths(text):
    for rx, kind in PATH_RULES:
        def repl(m, kind=kind):
            if kind == "t5root":
                rest = norm(m.group(1)).strip("/")
                return rest if rest else "t5-form-cost/"
            if kind == "local-rest":
                rest = norm(m.group(1)).strip("/")
                return "<local>/" + rest if rest else "<local>"
            if kind == "local-basename":
                rest = norm(m.group(1)).strip("/")
                return "<local>/" + rest.rsplit("/", 1)[-1] if rest else "<local>"
            return kind
        text = rx.sub(repl, text)
    return text


def fix_codes(text):
    for rx, new in RUNCODE_RULES:
        text = rx.sub(new, text)
    return text


def redact(text):
    for old, new in REDACT.items():
        text = text.replace(old, new)
    return text


def fix_string(value):
    return redact(fix_codes(fix_paths(value)))


def is_stamp(line):
    s = line.strip()
    return (s.startswith("Machine:") or re.search(r"\bOperator:\s", s) is not None
            or any(old in s for old in REDACT if new_is_label(old)))


def new_is_label(old):
    """Lines carrying the producer machine name are provenance stamps in prose files."""
    return REDACT[old] == PRODUCER


def wiki(text, exported):
    def repl(m):
        target, label = m.group(1).strip(), m.group(2)
        clean = target
        while clean.startswith("../"):
            clean = clean[3:]
        candidates = [clean, clean + ".md", clean + ".json"]
        hit = next((c for c in candidates if c in exported), None)
        if hit:
            return f"[{label or Path(clean).name}]({hit})"
        if label:
            return label
        return f"{Path(clean).name} (not exported)"
    return re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", repl, text)


def sanitize_markdown(text, exported):
    lines = text.splitlines(keepends=True)
    out, in_fm = [], False
    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_fm = True
            out.append(line)
            continue
        if in_fm:
            if line.strip() == "---":
                in_fm = False
            elif line.split(":", 1)[0].strip() in FRONTMATTER_DROP:
                continue
        if is_stamp(line):
            continue  # provenance stamp; blank lines left behind are collapsed below
        out.append(line)
    text = "".join(out)
    text = re.sub(r"(---\r?\n)(\r?\n){2,}", r"\1\2", text)
    return fix_string(wiki(text, exported))


def sanitize_python(text):
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        if line.strip().startswith("Machine:"):
            continue  # provenance stamp inside a module docstring
        out.append(line)
    return fix_string("".join(out))  # the producer constant is replaced by --redact


def drop_operator_lines(text):
    """Remove `"Operator": "<name>"` members at text level, keeping commas valid."""
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        if re.fullmatch(r'\s*"Operator":\s*"[^"]*",?\s*', line):
            if not line.rstrip().endswith(",") and out:
                prev = out[-1]
                body = prev.rstrip("\r\n")
                if body.endswith(","):
                    out[-1] = body[:-1] + prev[len(body):]
            continue
        out.append(line)
    return "".join(out)


def transform_obj(value):
    if isinstance(value, dict):
        return {fix_string(k): transform_obj(v) for k, v in value.items() if k != "Operator"}
    if isinstance(value, list):
        return [transform_obj(v) for v in value]
    if isinstance(value, str):
        return fix_string(value)
    return value


def sanitize_json(text):
    new = fix_string(drop_operator_lines(text))
    # proof: the text-level rewrite equals the declared object-level transform
    expected = transform_obj(json.loads(text))
    actual = json.loads(new)
    if expected != actual:
        raise SystemExit("JSON sanitization changed more than the declared fields")
    return new


def file_list(src):
    files = list(EXPLICIT)
    for pattern in RAW_GLOBS:
        files += sorted(str(f.relative_to(src)).replace("\\", "/") for f in src.glob(pattern))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--redact", action="append", default=[], metavar="OLD=NEW",
                    help="replace a private literal everywhere; NEW=producer-node marks the machine name")
    args = ap.parse_args(argv)
    for item in args.redact:
        old, sep, new = item.partition("=")
        if not sep or not old:
            raise SystemExit(f"--redact needs OLD=NEW: {item}")
        REDACT[old] = new
    if PRODUCER not in REDACT.values():
        raise SystemExit("pass --redact <producer machine name>=" + PRODUCER)
    src, dst = Path(args.source).resolve(), Path(args.target).resolve()
    if dst == src or src in dst.parents:
        raise SystemExit("target must be outside the source")
    files = file_list(src)
    exported = set(files) | {"README.md", "SANITIZATION.md"}
    rows = []
    for rel in files:
        path = src / rel
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise SystemExit(f"over 5 MB, review manually: {rel} {len(raw)}")
        text = raw.decode("utf-8")
        if rel.endswith(".md"):
            new = sanitize_markdown(text, exported)
        elif rel.endswith(".py"):
            new = sanitize_python(text)
        elif rel.endswith(".json"):
            new = sanitize_json(text)
        else:
            raise SystemExit(f"unexpected type: {rel}")
        data = new.encode("utf-8")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        rows.append((rel, sha(raw), sha(out.read_bytes()), len(data)))
    for rel, a, b, size in rows:
        print(f"{rel}\t{a}\t{b}\t{size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

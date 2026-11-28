"""Print one version's section of CHANGELOG.md (the GitHub release notes). Usage: release_notes.py X.Y.Z [changelog]"""
import os, re, sys


def notes(text, version):
    """The lines under '## <version>' up to the next '## ' heading; SystemExit if that version has no section."""
    m = re.search(rf"^## {re.escape(version)}(?=[ \t]|$).*?$\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m or not m.group(1).strip(): raise SystemExit(f"CHANGELOG.md has no notes under '## {version}'")
    return unwrap(m.group(1).strip()) + "\n"


def version(text):
    """The newest released version in the changelog (first '## X.Y.Z' heading), plus '+dev' when the Unreleased
    section has notes, i.e. this copy has changes made after that release."""
    m = re.search(r"^## (\d+\.\d+\.\d+)\b", text, re.M)
    v = m.group(1) if m else "0.0.0"
    u = re.search(r"^## Unreleased[ \t]*$\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return v + ("+dev" if u and u.group(1).strip() else "")


def unwrap(md):
    """Join wrapped lines (an indented continuation of a bullet, or the next line of a paragraph) onto the line they
    continue: GitHub release notes show every newline as a line break, so 120-column wrapping would show up."""
    out = []
    for line in md.split("\n"):
        cont = out and out[-1].strip() and line.strip() and not line.lstrip().startswith(("- ", "* ", "#")) and not out[-1].startswith("#")
        if cont: out[-1] += " " + line.strip()
        else: out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    v = sys.argv[1].removeprefix("ink-prospector-v").removeprefix("v")
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHANGELOG.md")
    sys.stdout.write(notes(open(path).read(), v))

"""Print one version's section of CHANGELOG.md (the GitHub release notes). Usage: release_notes.py X.Y.Z [changelog]"""
import os, re, sys


def notes(text, version):
    """The lines under '## <version>' up to the next '## ' heading; SystemExit if that version has no section."""
    m = re.search(rf"^## {re.escape(version)}(?=[ \t]|$).*?$\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m or not m.group(1).strip(): raise SystemExit(f"CHANGELOG.md has no notes under '## {version}'")
    return m.group(1).strip() + "\n"


if __name__ == "__main__":
    v = sys.argv[1].removeprefix("ink-prospector-v").removeprefix("v")
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHANGELOG.md")
    sys.stdout.write(notes(open(path).read(), v))

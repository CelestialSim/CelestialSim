# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate THIRD_PARTY_LICENSES_DEPS.md from the resolved Cargo graph.

Walks `cargo metadata` (the full transitive graph from Cargo.lock), splits
packages into RUNTIME-LINKED (compiled into the shipped cdylib) vs BUILD-TIME
ONLY (proc-macros / codegen — never in the binary, no binary-distribution
obligation), and emits one file with:

  * a table of every external package at any depth, with its license and
    linkage class;
  * the full license text + copyright lines for every runtime-linked crate
    (electing MIT from dual/triple expressions; MPL-2.0 crates get a section
    3.2-style source notice instead of the 373-line license body).

Exits non-zero if any strong-copyleft license (GPL/LGPL/AGPL/SSPL/BUSL/EUPL)
appears ANYWHERE in the graph, so CI fails before such a dependency ships.

Usage:  python3 tools/gen_dep_licenses.py          (regenerates the file)
CI:     regenerate, then `git diff --exit-code THIRD_PARTY_LICENSES_DEPS.md`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "THIRD_PARTY_LICENSES_DEPS.md"
COPYLEFT = ("GPL-", "GPL ", "AGPL", "LGPL", "SSPL", "BUSL", "EUPL")
# Preference order when a crate offers a choice of licenses.
ELECTION = ["MIT", "Zlib", "Apache-2.0", "MPL-2.0"]
LICENSE_FILES = {
    "MIT": ["LICENSE-MIT", "LICENSE-MIT.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"],
    "Zlib": ["LICENSE-ZLIB", "LICENSE-ZLIB.md", "LICENSE", "LICENSE.md"],
    "Apache-2.0": ["LICENSE-APACHE", "LICENSE-APACHE.md", "LICENSE"],
}


def is_copyleft(expr: str) -> bool:
    up = expr.upper()
    return any(tok in up for tok in COPYLEFT)


def elect(expr: str) -> str:
    """Pick the license we redistribute under from an SPDX OR-expression."""
    options = [t.strip("() ") for t in re.split(r"\bOR\b|/", expr)]
    for pref in ELECTION:
        for opt in options:
            # match "MIT" but also "(MIT" etc.; AND-riders stay attached.
            if opt.split(" AND ")[0].strip() == pref:
                return pref
    return options[0]


def license_text(pkg: dict, elected: str) -> tuple[str, str]:
    """Return (copyright_lines, full_text) for the crate's elected license."""
    root = Path(pkg["manifest_path"]).parent
    for name in LICENSE_FILES.get(elected, ["LICENSE", "LICENSE.md", "COPYING"]):
        p = root / name
        if p.is_file():
            text = p.read_text(errors="replace").strip()
            crs = [l.strip() for l in text.splitlines()
                   if l.strip().lower().startswith("copyright")][:3]
            return ("; ".join(crs) or f"the {pkg['name']} authors", text)
    return (f"the {pkg['name']} authors",
            f"(No license file bundled in the crate; the canonical {elected} license applies.)")


def main() -> int:
    meta = json.loads(
        subprocess.run(["cargo", "metadata", "--format-version", "1"],
                       capture_output=True, text=True, check=True).stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    ws = set(meta["workspace_members"])
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}

    def is_proc_macro(pid: str) -> bool:
        return any("proc-macro" in t["kind"] for t in packages[pid]["targets"])

    # Runtime-linked set: BFS from workspace members over NORMAL dep edges,
    # never entering (or traversing through) proc-macro crates.
    runtime: set[str] = set()
    queue = [m for m in ws]
    seen = set(queue)
    while queue:
        pid = queue.pop()
        for dep in nodes[pid]["deps"]:
            if not any(k["kind"] is None for k in dep["dep_kinds"]):
                continue  # build- or dev-only edge
            did = dep["pkg"]
            if did in ws or is_proc_macro(did):
                continue
            if did not in seen:
                seen.add(did)
                queue.append(did)
            runtime.add(did)

    external = sorted((p for pid, p in packages.items() if pid not in ws),
                      key=lambda p: p["name"])

    offenders = [p["name"] for p in external if p.get("license") and is_copyleft(p["license"])]
    if offenders:
        print(f"ERROR: strong-copyleft license in dependency graph: {offenders}", file=sys.stderr)
        return 1

    out = []
    out.append("# Third-party licenses — Cargo dependencies")
    out.append("")
    out.append("<!-- GENERATED FILE - do not edit. Regenerate with:")
    out.append("     python3 tools/gen_dep_licenses.py -->")
    out.append("")
    out.append("Licenses of the Rust crates this project depends on, from the full")
    out.append("transitive graph in `Cargo.lock`. **Runtime-linked** crates are compiled")
    out.append("into the distributed binaries (`libcelestialsim.{so,dll,dylib}`); their")
    out.append("license texts are reproduced below, as their terms require for binary")
    out.append("distribution. **Build-time only** crates (proc-macros / codegen) never")
    out.append("enter the binaries and carry no binary-distribution obligation; they are")
    out.append("listed for completeness. Hand-written third-party notices (vendored code,")
    out.append("shaders, assets) live in `THIRD_PARTY_NOTICES.md`.")
    out.append("")
    out.append("| Crate | Version | License | Linkage |")
    out.append("|---|---|---|---|")
    for p in external:
        linkage = "**runtime-linked**" if p["id"] in runtime else "build-time only"
        out.append(f"| {p['name']} | {p['version']} | {p['license']} | {linkage} |")
    out.append("")
    out.append("---")

    mpl_done = False
    for p in external:
        if p["id"] not in runtime:
            continue
        elected = elect(p["license"])
        if elected == "MPL-2.0":
            if mpl_done:
                continue
            mpl_crates = sorted(q["name"] for q in external
                                if q["id"] in runtime and elect(q["license"]) == "MPL-2.0")
            out.append("")
            out.append(f"## MPL-2.0 — {', '.join(mpl_crates)} (godot-rust/gdext)")
            out.append("")
            out.append("These crates are covered by the Mozilla Public License 2.0")
            out.append("(<https://mozilla.org/MPL/2.0/>). Per MPL-2.0 section 3.2, the")
            out.append("Corresponding Source is the `godot-rust/gdext` repository, used")
            out.append("unmodified at the commit pinned in this project's `Cargo.lock`:")
            out.append("<https://github.com/godot-rust/gdext>. You may obtain, modify and")
            out.append("redistribute that code under the terms of the MPL.")
            mpl_done = True
            continue
        copyright_line, text = license_text(p, elected)
        out.append("")
        out.append(f"## {elected} — {p['name']} {p['version']}")
        out.append("")
        out.append(f"{copyright_line}")
        out.append("")
        out.append("```")
        out.append(text)
        out.append("```")

    OUT.write_text("\n".join(out) + "\n")
    n_rt = sum(1 for p in external if p["id"] in runtime)
    print(f"wrote {OUT.name}: {len(external)} packages ({n_rt} runtime-linked), no copyleft")
    return 0


if __name__ == "__main__":
    sys.exit(main())

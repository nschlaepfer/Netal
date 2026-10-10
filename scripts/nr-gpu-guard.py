#!/usr/bin/env python3
"""Refuse to start a Mac GPU job while the shared journal's most recent GPU reservation belongs to another agent.
Scans docs/NR_AGENT_COORDINATION.md headings for RESERVE/RESERVED and RELEASE/RELEASED (Mac); the latest such heading decides.
Exit 0 = free or reserved by --me; exit 3 = reserved by someone else (prints the heading)."""
import argparse, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    p = argparse.ArgumentParser(); p.add_argument("--me", default="Fable"); p.add_argument("--journal", default=os.path.join(ROOT, "docs/NR_AGENT_COORDINATION.md")); a = p.parse_args()
    latest = {}   # agent -> ("reserved"|"free", ts, title): each agent's own most recent reservation state
    for line in open(a.journal, encoding="utf-8"):
        if not line.startswith("### "): continue
        m = re.match(r"### (\S+ \S+ UTC) — (\w+) — (.*)", line.strip())
        if not m: continue
        ts, who, title = m.groups(); t = title.upper()
        rel = [x.start() for x in re.finditer(r"\bRELEASED?\b", t)]; res = [x.start() for x in re.finditer(r"\bRESERVED?\b", t)]   # whole words: "preserved" is not a reservation
        has_rel, has_res = bool(rel), bool(res)
        if has_rel and has_res: latest[who] = ("reserved" if max(res) > max(rel) else "free", ts, title)
        elif has_rel: latest[who] = ("free", ts, title)
        elif has_res: latest[who] = ("reserved", ts, title)
    others = [(who, st) for who, st in latest.items() if who != a.me and st[0] == "reserved"]
    if others:
        for who, st in others: print(f"GPU reserved by {who} at {st[1]}: {st[2]}")
        sys.exit(3)
    print("GPU free (no other agent holds a reservation)" if latest.get(a.me, ("free",))[0] == "free" else f"GPU reserved by {a.me}"); sys.exit(0)
if __name__ == "__main__": main()

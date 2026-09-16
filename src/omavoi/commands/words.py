"""The two word lists: terms the model mishears, and proper nouns.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import config
from ..term import BOLD, DIM, GREEN, RED, RESET, YELLOW


def cmd_dict(args: argparse.Namespace) -> int:
    cfg = config.load()
    dictionary: dict[str, str] = dict(cfg.setdefault("dictionary", {}).get("rules", {}))
    if args.action == "list":
        if args.json:
            order = sorted(dictionary, key=len, reverse=True)
            rows = []
            for i, src in enumerate(order):
                shadow = next((o for o in order[:i] if o.lower() in src.lower()
                               or src.lower().startswith(o.lower())), "")
                rows.append({"heard": src, "meant": dictionary[src], "shadowed_by": shadow})
            print(json.dumps({"rules": rows}, ensure_ascii=False, indent=2))
            return 0
        for src in sorted(dictionary, key=len, reverse=True):
            print(f"  {src}  ->  {dictionary[src]}")
        print(f"\n{DIM}{len(dictionary)} entries{RESET}")
        return 0
    if args.action == "add":
        if not args.heard or not args.meant:
            print(f"{RED}usage: omavoi dict add <heard> <meant>{RESET}", file=sys.stderr)
            return 1
        dictionary[args.heard] = args.meant
        cfg["dictionary"]["rules"] = dictionary
        config.write(cfg)
        print(f"{GREEN}ok{RESET} {args.heard} -> {args.meant}  "
              f"{DIM}(omavoi reload to apply){RESET}")
        return 0
    if args.action == "rm":
        if dictionary.pop(args.heard, None) is None:
            print(f"{DIM}{args.heard} is not in the dictionary{RESET}")
            return 1
        cfg["dictionary"]["rules"] = dictionary
        config.write(cfg)
        print(f"{GREEN}removed{RESET} {args.heard}")
        return 0
    return 1


def cmd_names(args: argparse.Namespace) -> int:
    from .. import names as names_mod

    cfg = config.load()
    entries = cfg.setdefault("dictionary", {}).setdefault("names", [])

    if args.action == "list":
        index = names_mod.NameIndex(cfg)
        if args.json:
            rows = []
            for e in index.entries:
                key = (names_mod.pinyin_key(e.name) if e.resolved_match() == "pinyin"
                       else names_mod.phonetic_key(e.name))
                rows.append(e.as_dict() | {"key": key})
            print(json.dumps({"names": rows, "seed": index.seed_text(),
                              "budget": index.budget}, ensure_ascii=False, indent=2))
            return 0
        if not index.entries:
            print(f"{DIM}no names yet — omavoi names add <name> [...]{RESET}")
            return 0
        for e in index.entries:
            state = f"{GREEN}matching{RESET}" if e.enabled else f"{DIM}seed only{RESET}"
            key = (names_mod.pinyin_key(e.name) if e.resolved_match() == "pinyin"
                   else names_mod.phonetic_key(e.name))
            print(f"  {e.name:<18}{key:<22}{e.resolved_match():<10}{state:<20}{DIM}{e.group}{RESET}")
        print(f"\n{DIM}{len(index.entries)} names · decoder prompt: {index.seed_text()[:60] or '(none)'}{RESET}")
        return 0

    if args.action == "add":
        if not args.names:
            print(f"{RED}usage: omavoi names add <name> [<name> ...]{RESET}", file=sys.stderr)
            return 1
        existing = {e.name for e in names_mod.load(cfg)}
        added = 0
        for name in args.names:
            name = name.strip()
            if not name or name in existing:
                continue
            entries.append({"name": name, "group": args.group, "seed": True, "enabled": False})
            added += 1
        config.write(cfg)
        print(f"{GREEN}added {added}{RESET} — seeded into the decoder prompt now.")
        print(f"{DIM}Sound matching stays off until `omavoi names dryrun` shows what it would do.{RESET}")
        return 0

    if args.action == "rm":
        target = set(args.names)
        kept = [e for e in entries if (e.get("name") if isinstance(e, dict) else e) not in target]
        removed = len(entries) - len(kept)
        cfg["dictionary"]["names"] = kept
        config.write(cfg)
        print(f"{GREEN}removed {removed}{RESET}")
        return 0

    if args.action in ("dryrun", "enable"):
        # Only this branch reads past takes. It was imported at the top of
        # the function, so `names list` and `names add` each pulled
        # history — and numpy behind it — to do nothing with either.
        from ..history import History

        texts = [e.get("raw_text") or "" for e in History(cfg).iter_entries()]
        texts = [t for t in texts if t.strip()]
        if not texts:
            print(f"{YELLOW}no stored transcripts to test against yet{RESET}")
            return 1
        found = names_mod.dry_run(cfg, texts)
        print(f"{BOLD}{len(found)}{RESET} of {len(texts)} stored takes would change\n")
        for item in found[:20]:
            print(f"  {DIM}{item['before']}{RESET}")
            print(f"  {GREEN}{item['after']}{RESET}")
            print(f"    {DIM}{', '.join(h['found'] + ' -> ' + h['name'] for h in item['hits'])}{RESET}\n")
        if args.action == "dryrun":
            print(f"{DIM}Look these over. `omavoi names enable` turns matching on for all of them.{RESET}")
            return 0
        for e in entries:
            if isinstance(e, dict):
                e["enabled"] = True
        config.write(cfg)
        print(f"{GREEN}sound matching enabled for all names{RESET}")
        return 0
    return 1

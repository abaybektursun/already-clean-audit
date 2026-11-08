"""Import answers written by Claude Code answering agents into a batch's responses file (manual rows of a
mixed batch, neurons_llm 2026-10-21).

usage: python3 manual_import.py <explain dir> <batch> <answers jsonl> [<answers jsonl> ...]

A row is imported when it has exactly the keys id, model, text, utc; its id is a request of the batch
with no response line and no decline; its model equals the request's model; its utc is a real time within the last
24 hours; and text parses as a JSON object. It is appended to responses/<batch>.jsonl with "backend": "manual". Every other row is listed
and left out (its id stays pending). Schema and role checks are `reject`'s job (--backend mixed).
"""
import json
import pathlib
import sys


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if pathlib.Path(p).exists() else []


def fresh(utc: str) -> bool:
    """True when utc parses as YYYY-MM-DDTHH:MM:SSZ and lies within the 24 hours before now (an agent's placeholder
    time would otherwise be logged as the answer's time and could predate the judge gate)."""
    import datetime
    try:
        t = datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return False
    now = datetime.datetime.now(datetime.timezone.utc)
    return datetime.timedelta(0) <= now - t <= datetime.timedelta(hours=24)


def main(explain, batch, files):
    explain = pathlib.Path(explain)
    requests = {r["id"]: r for r in read_jsonl(explain / "requests" / f"{batch}.jsonl")}
    resp_path = explain / "responses" / f"{batch}.jsonl"
    done = {r["id"] for r in read_jsonl(resp_path)}
    done |= {r["id"] for r in read_jsonl(explain / "responses" / f"{batch}.declined.jsonl")}
    good, bad = [], []
    for f in files:
        for n, line in enumerate(open(f), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append((f, n, None, f"line is not JSON ({e.msg})")); continue
            rid = row.get("id")
            why = None
            if sorted(row) != ["id", "model", "text", "utc"]:
                why = f"keys {sorted(row)}"
            elif rid not in requests:
                why = "no such request"
            elif rid in done:
                why = "already answered or declined"
            elif row["model"] != requests[rid]["model"]:
                why = f"model {row['model']!r} != request's {requests[rid]['model']!r}"
            elif not fresh(row["utc"]):
                why = f"utc {row['utc']!r} is not within the last 24 hours (placeholder or stale time)"
            else:
                try:
                    obj = json.loads(row["text"])
                    if not isinstance(obj, dict):
                        why = "text is not a JSON object"
                except json.JSONDecodeError as e:
                    why = f"text is not JSON ({e.msg} at char {e.pos})"
            if why:
                bad.append((f, n, rid, why)); continue
            done.add(rid)
            good.append({**row, "backend": "manual"})
    with open(resp_path, "a") as out:
        for row in good:
            out.write(json.dumps(row) + "\n")
    for f, n, rid, why in bad:
        print(f"left out {f}:{n} {rid}: {why}")
    print(json.dumps({"batch": batch, "imported": len(good), "left_out": len(bad), "responses": str(resp_path)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

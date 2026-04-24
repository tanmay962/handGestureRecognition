#!/usr/bin/env python3
"""
Pre-demo readiness check for the gesture detection model.

Simulates samples for every trained gesture, runs them through the live
prediction endpoint, and prints a colour-coded GO / WARN / FAIL report.

Usage:
    python demo_validator.py                   # full readiness report
    python demo_validator.py --live            # tail live predictions
    python demo_validator.py --gesture Stop    # spot-check one gesture
    python demo_validator.py --samples 30      # use 30 simulated samples
    python demo_validator.py --url http://localhost:8000
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict

# Config

DEFAULT_URL     = "http://localhost:8000"
SAMPLES_DEFAULT = 25

PASS_ACC   = 0.80
PASS_CONF  = 0.75
WARN_ACC   = 0.60
WARN_CONF  = 0.60

# ANSI colours

def _c(code, text):
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("1;92", t)
YELLOW = lambda t: _c("1;93", t)
RED    = lambda t: _c("1;91", t)
CYAN   = lambda t: _c("96",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)


# HTTP helpers

def _request(method, url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_msg": e.reason}
    except Exception as e:
        return {"_error": str(e)}

def api_get(base, path):
    return _request("GET", base + path)

def api_post(base, path, body):
    return _request("POST", base + path, body)


# Display helpers

def hr(char="─", width=64):
    print(DIM(char * width))

def section(title):
    print()
    hr("═")
    print(BOLD(f"  {title}"))
    hr("═")

def badge(acc, conf):
    if acc >= PASS_ACC and conf >= PASS_CONF:
        return GREEN(" GO   ")
    if acc >= WARN_ACC and conf >= WARN_CONF:
        return YELLOW(" WARN ")
    return RED(" FAIL ")

def bar(fraction, width=20):
    filled = int(fraction * width)
    b = "█" * filled + "░" * (width - filled)
    pct = f"{fraction*100:5.1f}%"
    if fraction >= PASS_ACC:
        return GREEN(b) + f" {pct}"
    if fraction >= WARN_ACC:
        return YELLOW(b) + f" {pct}"
    return RED(b) + f" {pct}"


# Checks

def check_server(base):
    print(f"\n{BOLD('Connecting to')} {CYAN(base)} …", end=" ", flush=True)
    resp = api_get(base, "/api/health")
    if "_error" in resp:
        print(RED("OFFLINE"))
        print(f"  {RED('✗')} Could not reach backend: {resp['_error']}")
        print("  → Start the server first:  cd backend && python main.py")
        sys.exit(1)
    print(GREEN("online"))

    status = api_get(base, "/api/nn/status")
    trained = status.get("static", {}).get("trained", False)
    acc     = status.get("static", {}).get("accuracy",     0.0)
    val_acc = status.get("static", {}).get("val_accuracy", 0.0)
    epochs  = status.get("static", {}).get("epochs",       0)

    print(f"  {GREEN('✓')} Backend healthy | device={resp.get('device','?')}")
    if not trained:
        print(f"  {RED('✗')} Model is NOT trained — train first then re-run this script.")
        sys.exit(1)
    print(f"  {GREEN('✓')} Model trained | epochs={epochs}  "
          f"train_acc={acc:.1%}  val_acc={val_acc:.1%}")
    return trained


def get_gesture_registry(base):
    rows = api_get(base, "/api/gestures")
    if isinstance(rows, list):
        return {r["name"]: r.get("gesture_type", "static") for r in rows}
    return {}


def test_gesture(base, name, gtype, n):
    sim = api_post(base, "/api/samples/simulate",
                   {"gesture": name, "count": n, "sample_type": gtype})

    samples = sim.get("samples", [])
    if not samples:
        return None

    hits      = 0
    confs     = []
    confusion = defaultdict(int)

    for sample in samples:
        res = api_post(base, "/api/nn/predict",
                       {"features": sample, "model_type": gtype})
        if "_error" in res:
            continue
        predicted = res.get("name", "Unknown")
        conf      = float(res.get("conf", 0.0))
        confs.append(conf)
        confusion[predicted] += 1
        if predicted == name:
            hits += 1

    total = len(confs)
    if total == 0:
        return None

    return {
        "name":      name,
        "gtype":     gtype,
        "total":     total,
        "hit_rate":  hits / total,
        "avg_conf":  sum(confs) / total,
        "min_conf":  min(confs),
        "confusion": dict(confusion),
    }


# Report modes

def full_report(base, n_samples, target_gesture=None):
    registry = get_gesture_registry(base)
    if not registry:
        print(RED("  No gestures registered."))
        sys.exit(1)

    if target_gesture:
        if target_gesture not in registry:
            print(RED(f"  Gesture '{target_gesture}' not found in registry."))
            sys.exit(1)
        registry = {target_gesture: registry[target_gesture]}

    section("GESTURE-BY-GESTURE READINESS REPORT")
    print(f"  Testing {len(registry)} gesture(s) with {n_samples} simulated samples each …\n")

    col_w = max(len(n) for n in registry) + 2

    header = (
        f"  {'Gesture':<{col_w}}  {'Type':<8}  {'Hit-rate':<30}  "
        f"{'Avg conf':>8}  {'Min conf':>8}  Status"
    )
    print(BOLD(header))
    hr()

    results      = []
    go_count     = 0
    warn_count   = 0
    fail_count   = 0
    confused_map = {}

    for name, gtype in registry.items():
        print(f"  {name:<{col_w}}  testing…", end="\r", flush=True)
        r = test_gesture(base, name, gtype, n_samples)

        if r is None:
            print(f"  {name:<{col_w}}  {DIM('(no samples — skipped)')}")
            continue

        status_badge = badge(r["hit_rate"], r["avg_conf"])
        print(
            f"  {name:<{col_w}}  {gtype:<8}  "
            f"{bar(r['hit_rate']):<30}  "
            f"{r['avg_conf']:>7.1%}  "
            f"{r['min_conf']:>7.1%}  "
            f"{status_badge}"
        )

        wrong = {k: v for k, v in r["confusion"].items() if k != name}
        if wrong:
            top_wrong = max(wrong, key=wrong.get)
            confused_map[name] = (top_wrong, wrong[top_wrong], r["total"])

        results.append(r)
        if   r["hit_rate"] >= PASS_ACC and r["avg_conf"] >= PASS_CONF: go_count   += 1
        elif r["hit_rate"] >= WARN_ACC and r["avg_conf"] >= WARN_CONF: warn_count += 1
        else:                                                            fail_count += 1

    section("SUMMARY")
    total_tested = go_count + warn_count + fail_count
    print(f"  {GREEN(f'GO    {go_count:>3}')}  gestures  (hit ≥ {PASS_ACC:.0%}, conf ≥ {PASS_CONF:.0%})")
    print(f"  {YELLOW(f'WARN  {warn_count:>3}')}  gestures  (hit ≥ {WARN_ACC:.0%}, conf ≥ {WARN_CONF:.0%})")
    print(f"  {RED(f'FAIL  {fail_count:>3}')}  gestures  (below WARN thresholds)")
    hr()

    if confused_map:
        print(f"\n  {BOLD('Confusion warnings')} (gestures mistaken for another):\n")
        for src, (dst, cnt, tot) in confused_map.items():
            pct = cnt / tot
            col = RED if pct > 0.3 else YELLOW
            print(f"  {col(f'{src:<{col_w}}')} → predicted as {BOLD(dst)} "
                  f"{cnt}/{tot} times ({pct:.0%})")

    print()
    if fail_count == 0 and warn_count == 0:
        print(f"  {GREEN('██████████████████████████████')}")
        print(f"  {GREEN('  DEMO READY — all gestures GO  ')}")
        print(f"  {GREEN('██████████████████████████████')}")
    elif fail_count == 0:
        print(f"  {YELLOW('⚠  MOSTLY READY')} — "
              f"{warn_count} gesture(s) may be shaky; rehearse them first.")
    else:
        print(f"  {RED('✗  NOT READY')} — "
              f"{fail_count} gesture(s) failing.  Capture more samples and retrain.")

    if results:
        section("RECOMMENDED DEMO ORDER  (highest confidence first)")
        ordered = sorted(results, key=lambda r: (r["hit_rate"] + r["avg_conf"]), reverse=True)
        for i, r in enumerate(ordered, 1):
            b = badge(r["hit_rate"], r["avg_conf"])
            print(f"  {i:>2}. {b}  {r['name']:<{col_w}}  "
                  f"hit={r['hit_rate']:.0%}  conf={r['avg_conf']:.0%}")

    print()


def live_monitor(base):
    section("LIVE PREDICTION MONITOR  (Ctrl+C to stop)")
    print(f"  Watching {CYAN(base + '/api/predictions/recent')} …\n")

    seen_ids  = set()
    last_name = None

    print(f"  {'Time':<10}  {'Gesture':<20}  {'Confidence':<28}  Type")
    hr()

    try:
        while True:
            preds = api_get(base, "/api/predictions/recent?limit=10")
            if not isinstance(preds, list):
                time.sleep(0.5)
                continue

            new_preds = [p for p in preds if p.get("id") not in seen_ids]
            for p in reversed(new_preds):
                seen_ids.add(p["id"])
                name  = p.get("gesture", "?")
                conf  = float(p.get("confidence", 0.0))
                mtype = p.get("model_type", "")
                ts    = time.strftime("%H:%M:%S", time.localtime(p.get("created_at", 0)))

                repeat = (name == last_name)
                label  = DIM(f"  ({name})") if repeat else GREEN(f"  ► {name}")
                conf_b = bar(conf, width=16)
                print(f"  {DIM(ts)}  {label:<28}  {conf_b}  {DIM(mtype)}")
                last_name = name

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n  {DIM('Monitor stopped.')}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-demo gesture model validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",      default=DEFAULT_URL,     help="Backend URL")
    parser.add_argument("--samples",  type=int, default=SAMPLES_DEFAULT,
                        help="Simulated samples per gesture (default 25)")
    parser.add_argument("--gesture",  default=None,
                        help="Test a single gesture instead of all")
    parser.add_argument("--live",     action="store_true",
                        help="Monitor live predictions in real time")
    args = parser.parse_args()

    print(BOLD("\n  ╔══════════════════════════════════════╗"))
    print(BOLD("  ║   Gesture Detection — Demo Validator  ║"))
    print(BOLD("  ╚══════════════════════════════════════╝"))

    check_server(args.url)

    if args.live:
        live_monitor(args.url)
    else:
        full_report(args.url, args.samples, args.gesture)


if __name__ == "__main__":
    main()

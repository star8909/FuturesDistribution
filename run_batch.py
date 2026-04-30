"""Batch iter runner — 8개 동시 실행.

사용:
    python run_batch.py iter162 iter163 iter164
    python run_batch.py --workers 4 iter150 iter151 iter152

자동 발견:
    python run_batch.py --all-pending  # iters/ 안에 있지만 logs/에 없는 것
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_iter(name: str) -> Path | None:
    """iter162 → iters/iter162_xxx.py 자동 매칭."""
    iters_dir = ROOT / "iters"
    matches = list(iters_dir.glob(f"{name}_*.py")) + list(iters_dir.glob(f"{name}.py"))
    if not matches:
        return None
    return matches[0]


def run_one(iter_path: str) -> dict:
    """Run a single iter, capture output to logs/<iter>.log."""
    p = Path(iter_path)
    log_path = ROOT / "logs" / f"{p.stem}.log"
    log_path.parent.mkdir(exist_ok=True)
    start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                [sys.executable, str(iter_path)],
                stdout=f, stderr=subprocess.STDOUT,
                env=env, cwd=str(ROOT),
                timeout=3600,
            )
        elapsed = time.time() - start
        return {
            "iter": p.stem,
            "status": "ok" if result.returncode == 0 else "fail",
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed, 1),
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {"iter": p.stem, "status": "timeout", "elapsed_sec": 3600, "log": str(log_path)}
    except Exception as e:
        return {"iter": p.stem, "status": "error", "error": str(e), "log": str(log_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iters", nargs="*", help="iter 이름 (e.g. iter162) 또는 경로")
    ap.add_argument("--workers", type=int, default=8, help="동시 실행 수 (default: 8)")
    ap.add_argument("--all-pending", action="store_true", help="결과 없는 모든 iter 자동 실행")
    args = ap.parse_args()

    if args.all_pending:
        all_iters = sorted((ROOT / "iters").glob("iter*.py"))
        results_dir = ROOT / "results"
        pending = []
        for ip in all_iters:
            name = ip.stem
            # results/<name>.json 또는 results/<name 일부>.json 확인
            if not any(results_dir.glob(f"{name.split('_')[0]}_*.json")):
                pending.append(str(ip))
        targets = pending
        print(f"자동 발견: {len(targets)}개 미실행 iter")
    else:
        targets = []
        for arg in args.iters:
            p = Path(arg)
            if p.is_file():
                targets.append(str(p))
            else:
                # iter 이름으로 검색
                found = find_iter(arg)
                if found:
                    targets.append(str(found))
                else:
                    print(f"⚠️ {arg} 못 찾음, skip")

    if not targets:
        print("❌ 실행할 iter 없음")
        return

    print(f"\n[batch] {len(targets)}개 iter, {args.workers} workers")
    for t in targets:
        print(f"  - {Path(t).stem}")

    start_all = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_to_iter = {ex.submit(run_one, t): t for t in targets}
        for future in as_completed(future_to_iter):
            res = future.result()
            results.append(res)
            mark = "✅" if res.get("status") == "ok" else "❌"
            print(f"  {mark} {res['iter']:30s} {res.get('status'):8s} {res.get('elapsed_sec', 0):.1f}s")

    elapsed_all = time.time() - start_all
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n=== 완료 ===")
    print(f"  성공: {n_ok}/{len(results)}")
    print(f"  총 시간: {elapsed_all:.1f}s ({elapsed_all/60:.1f}분)")
    print(f"  평균/iter: {elapsed_all/len(results):.1f}s")
    print(f"  speedup vs 직렬: {sum(r.get('elapsed_sec', 0) for r in results) / elapsed_all:.1f}x")


if __name__ == "__main__":
    main()

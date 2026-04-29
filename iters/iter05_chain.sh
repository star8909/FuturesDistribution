#!/usr/bin/env bash
# iter02 (Agri 안전형) 끝나면 iter03 → iter04 자동 chain 실행
# 사용: bash iters/iter05_chain.sh > logs/chain.log 2>&1 &

cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

# iter02 완료 대기 (json 생성 시까지)
echo "Waiting for iter02 completion..."
while [ ! -f results/iter02_agri_safe.json ]; do sleep 30; done
echo "iter02 done. Starting iter03 vol-target..."

python iters/iter03_voltarget.py > logs/iter03.log 2>&1
echo "iter03 done. Starting iter04 equity stop..."

python iters/iter04_equity_stop.py > logs/iter04.log 2>&1
echo "iter04 done. All chain finished."

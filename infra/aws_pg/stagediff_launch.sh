#!/usr/bin/env bash
# STAGEDIFF wave launcher — every cell goes through `pg run` with a fresh S3 prefix, `--weights` naming only
# its model dir, `--budget-min`, `--json --detach`; a launch that hits the vCPU quota (4 × g6e.2xlarge) is
# retried every 60 s, which queues the excess cells behind the running ones. Usage (from the worktree root):
#   bash infra/aws_pg/stagediff_launch.sh w1 <attempt> [--bos-ablation]
#   bash infra/aws_pg/stagediff_launch.sh w2 <attempt> [dim_batch]
#   bash infra/aws_pg/stagediff_launch.sh w3 <attempt> <hop> <parent-fit-prefix[,prefix2]> <child-fit-prefix>
#   bash infra/aws_pg/stagediff_launch.sh w7 <attempt>            (SINKPATCH; W7_HOPS="H5p" to launch one)
#   bash infra/aws_pg/stagediff_launch.sh poll <wave> <attempt> <cell> [cell ...]
set -uo pipefail
B=pg-lineage-769063704696
PG=./infra/aws_pg/pg
# macOS ships bash 3.2 (no associative arrays): plain case functions
wdir() { case "$1" in
  C0) echo models--Qwen--Qwen3-4B-Instruct-2507;; C1) echo models--allenai--Molmo2-4B;; C2) echo models--allenai--Molmo2-ER;;
  C3) echo models--allenai--MolmoAct2-Pretrain;; C4) echo models--allenai--MolmoAct2;; C5) echo models--allenai--MolmoAct2-LIBERO;;
  C6) echo models--allenai--MolmoAct2-DROID;; P0) echo models--Qwen--Qwen3-4B;; *) echo "unknown ckpt $1" >&2; exit 2;; esac; }
hopp() { case "$1" in H1) echo C0;; H2) echo C1;; H3) echo C2;; H4) echo C3;; H5|H5p) echo C4;; *) exit 2;; esac; }
hopc() { case "$1" in H1) echo C1;; H2) echo C2;; H3) echo C3;; H4) echo C4;; H5) echo C5;; H5p) echo C6;; *) exit 2;; esac; }

launch() {   # launch <name> <pg run args...>
  local name="$1"; shift
  for t in $(seq 1 120); do
    out=$($PG run "$@" --json --detach 2>&1 | tail -1)
    if echo "$out" | grep -q instance_id; then echo "$(date -u +%H:%M:%SZ) launched $name: $out"; return 0; fi
    echo "$(date -u +%H:%M:%SZ) $name waiting for a slot ($t): $(echo "$out" | tail -c 120)"; sleep 60
  done
  echo "GAVE UP $name"; return 1
}

cmd="${1:-}"; shift || true
case "$cmd" in
  w1)
    att="$1"; shift; extra="${1:-}"
    for c in ${W1_CELLS:-C0 C1 C2 C3 C4 C5 C6}; do
      budget=45; [ "$c" = C1 ] || [ "$c" = C2 ] && budget=55
      launch "bat-$c" infra.aws_pg.jobs.stagediff_battery --args " --ckpt $c $extra" --budget-min $budget \
        --weights "$(wdir $c)" --s3-prefix "results/stagediff/w1/bat-$c/$att"
    done ;;
  w2)
    att="$1"; shift; db="${1:-32}"
    for half in a b; do
      launch "fit-C0$half" infra.aws_pg.jobs.stagediff_fit --args " --ckpt C0 --corpus-half $half --dim-batch $db" --budget-min 100 \
        --weights "$(wdir C0)" --s3-prefix "results/stagediff/w2/fit-C0$half/$att"
    done
    for c in ${W2_CELLS:-C1 C2 C3 C4 C5 C6}; do
      budget=150; [ "$c" = C1 ] || [ "$c" = C2 ] && budget=160
      launch "fit-$c" infra.aws_pg.jobs.stagediff_fit --args " --ckpt $c --dim-batch $db" --budget-min $budget \
        --weights "$(wdir $c)" --s3-prefix "results/stagediff/w2/fit-$c/$att"
    done ;;
  w3)
    att="$1"; hop="$2"; pf="$3"; cf="$4"
    p="$(hopp $hop)"; c="$(hopc $hop)"
    launch "pair-$hop" infra.aws_pg.jobs.stagediff_pair --args " --hop $hop --parent-fit $pf --child-fit $cf" --budget-min 120 \
      --weights "$(wdir $p),$(wdir $c)" --s3-prefix "results/stagediff/w3/pair-$hop/$att" ;;
  w6)   # review plan 2026-09-05: E3 (i2m-H5, i2m-H5p) first (longest), then E2 (rot-floor-C0), then E1 (rank-C0..C6)
    # 14:34Z: g6e.2xlarge/4xlarge InsufficientInstanceCapacity in every AZ → g5.2xlarge (A10G 24 GB, a different pool) as
    # the last fallback; these jobs only run forwards on a 4B bf16 model, no lens fit; budgets sized for the slower GPU.
    # 14:41Z: g5.2xlarge dry too → every single-GPU type the DLAMI supports, fastest first; budgets cover the slowest (L4).
    att="$1"; F="s3://$B/results/stagediff/w2"; TYPES="${W6_TYPES:-g6e.2xlarge,g6e.4xlarge,g6e.xlarge,g5.2xlarge,g5.4xlarge,g5.xlarge,g6.2xlarge,g6.4xlarge}"
    for hop in ${W6_I2M:-H5 H5p}; do
      c="$(hopc $hop)"; p="$(hopp $hop)"
      launch "i2m-$hop" infra.aws_pg.jobs.stagediff_probe --args " --ckpt $c --hop $hop --parent-fit $F/fit-$p/1/out --child-fit $F/fit-$c/1/out --i2-matched --trace-fullvocab" \
        --budget-min 100 --type "$TYPES" --weights "$(wdir $c)" --s3-prefix "results/stagediff/w6/i2m-$hop/$att"
    done
    launch "rot-floor-C0" infra.aws_pg.jobs.stagediff_rotfloor --args " --ckpt C0 --fit-a $F/fit-C0a/1/out --fit-b $F/fit-C0b/1/out" \
      --budget-min 35 --type "$TYPES" --weights "$(wdir C0)" --s3-prefix "results/stagediff/w6/rot-floor-C0/$att"
    for c in ${W6_RANK:-C0 C1 C2 C3 C4 C5 C6}; do
      budget=35; [ "$c" = C1 ] || [ "$c" = C2 ] && budget=45
      launch "rank-$c" infra.aws_pg.jobs.stagediff_rank --args " --ckpt $c" --budget-min $budget --type "$TYPES" \
        --weights "$(wdir $c)" --s3-prefix "results/stagediff/w6/rank-$c/$att"
    done ;;
  w7)   # SINKPATCH (studies/sinkpatch/SPEC.md §9): one cell per hop, parent and child co-resident; H5p first, then H5, H4
    att="$1"; shift; TYPES="${W7_TYPES:-g6e.2xlarge,g6e.4xlarge,g6e.xlarge,g5.2xlarge,g5.4xlarge,g5.xlarge,g6.2xlarge,g6.4xlarge}"
    for hop in ${W7_HOPS:-H5p H5 H4}; do
      c="$(hopc $hop)"; p="$(hopp $hop)"
      launch "sink-$hop" infra.aws_pg.jobs.sinkpatch_hop --args " --hop $hop" --budget-min "${W7_BUDGET:-45}" --type "$TYPES" \
        --weights "$(wdir $p),$(wdir $c)" --s3-prefix "results/stagediff/w7/sink-$hop/$att"
    done ;;
  poll)
    wave="$1"; att="$2"; shift 2; cells=("$@")
    for i in $(seq 1 240); do
      n=0; line=""
      for c in "${cells[@]}"; do
        s=$(aws s3 cp "s3://$B/results/stagediff/$wave/$c/$att/STATUS" - --profile fuelos --region us-east-1 2>/dev/null)
        if [ -n "$s" ]; then n=$((n+1)); line="$line $c:$s"; else line="$line $c:-"; fi
      done
      echo "$(date -u +%H:%M:%SZ) $n/${#cells[@]} $line"
      [ "$n" -eq "${#cells[@]}" ] && break
      running=$($PG ps 2>/dev/null | grep -c "running\|pending")
      if [ "$running" -eq 0 ]; then echo "no instances running but only $n STATUS files"; break; fi
      sleep 60
    done ;;
  *) echo "usage: $0 w1|w2|w3|poll ..."; exit 2 ;;
esac

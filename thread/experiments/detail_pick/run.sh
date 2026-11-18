#!/bin/sh
# experiments/detail_pick/run.sh LABEL "WORLD=BUDGET ..." [VAR=VALUE ...]     (from thread/)
# Paint to a plan in a copy of each world and refinish it under LABEL. The copy (evals/work/dp/LABEL/WORLD) keeps the
# world's own pictures of the usual views: only the detail views are painted. BUDGET: the time to paint, in ordinary
# pictures (what pipeline.py hands chisel_ring.py prep as WORLD_PAINT_VIEWS). VAR=VALUE: environment for every step.
label=$1; worlds=$2; shift 2
for kv in "$@"; do export "$kv"; done
for wb in $worlds; do
  w=${wb%=*}; b=${wb#*=}; d=evals/work/dp/$label/$w
  if [ -n "$KEEP_PAINT" ] && [ -f $d/paint.log ]; then      # KEEP_PAINT=1: a copy painted already is only refinished again
    .venv/bin/python evals/quick.py refinish $d --label $label > $d/refinish.json 2> $d/refinish.err || { echo "$w refinish FAILED"; tail -5 $d/refinish.err; }
    echo "$w: refinished again"; continue
  fi
  rm -rf $d; mkdir -p $d/shapes
  cp worlds/$w/boxes.json $d/; cp -R worlds/$w/gen $d/gen; [ -d worlds/$w/shapes/placed ] && cp -R worlds/$w/shapes/placed $d/shapes/placed; [ -f worlds/$w/shapes/plan.json ] && cp worlds/$w/shapes/plan.json $d/shapes/
  v=$(.venv/bin/python -c "import json;v=json.load(open('$d/gen/views.json'));print(v['views'],'%dx%d'%tuple(v['paint']))")
  style=$(.venv/bin/python -c "import json;print(json.load(open('$d/gen/views.json'))['style'])")
  export WORLD_DIR=$d WORLD_VIEWS=${v% *} WORLD_PAINT=${v#* }
  s=$(date +%s); WORLD_PAINT_VIEWS=$b WORLD_DETAIL_PLAN=1 WORLD_DETAIL_REPLAN=1 .venv/bin/python chisel_ring.py prep > $d/prep.log 2>&1 || { echo "$w prep FAILED"; tail -5 $d/prep.log; continue; }
  p=$(date +%s); ../spike/.venv/bin/python chisel_ring.py gen gen "$style" > $d/paint.log 2>&1 || { echo "$w paint FAILED"; tail -5 $d/paint.log; continue; }
  e=$(date +%s); echo "$w: prep $((p - s)) s, paint $((e - p)) s; $(grep 'time for' $d/prep.log)"; grep "carried over" $d/paint.log | tr '\n' ';'; echo
  unset WORLD_DIR WORLD_VIEWS WORLD_PAINT
  .venv/bin/python evals/quick.py refinish $d --label $label > $d/refinish.json 2> $d/refinish.err || { echo "$w refinish FAILED"; tail -5 $d/refinish.err; }
done

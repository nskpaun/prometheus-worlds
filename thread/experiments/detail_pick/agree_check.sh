#!/bin/sh
# experiments/detail_pick/agree_check.sh  (from thread/): a world painted without a plan must get the views of the
# previous stage (label gaps7) bit for bit from the agree step
for w in ${@:-ex-tower garage-shapes ex-blossom}; do
  d=evals/work/dp/agree-$w; rm -rf $d; mkdir -p $d/shapes
  cp worlds/$w/boxes.json $d/; cp -R worlds/$w/gen $d/gen; [ -d worlds/$w/shapes/placed ] && cp -R worlds/$w/shapes/placed $d/shapes/placed; [ -f worlds/$w/shapes/plan.json ] && cp worlds/$w/shapes/plan.json $d/shapes/
  v=$(.venv/bin/python -c "import json;v=json.load(open('$d/gen/views.json'));print(v['views'],'%dx%d'%tuple(v['paint']))")
  s=$(date +%s); WORLD_ROOM_ONLY=1 WORLD_DIR=$d WORLD_VIEWS=${v% *} WORLD_PAINT=${v#* } .venv/bin/python consistent.py gen > $d/agree.log 2>&1 || { echo "$w FAILED"; tail -5 $d/agree.log; continue; }
  a=$(cd $d/gen_c/images && cat *.png | md5); b=$(cd evals/work/$w-gaps7/gen_c/images && cat *.png | md5)
  [ "$a" = "$b" ] && echo "$w same as gaps7 ($(ls $d/gen_c/images | wc -l | tr -d ' ') views, $(( $(date +%s) - s )) s)" || echo "$w DIFFERS"
done

#!/bin/sh
# experiments/gaps/paint_lookdown.sh WORLD...   (from thread/)
# A copy of each world in evals/work/ld/WORLD with its downward views placed by the plan (chisel_ring.plan_lookdown) and
# only those painted again: the ring's pictures are the world's own. The views after the ring are painted last and
# carry the ring's paint over, so painting just them is what a whole build would do after the ring.
for w in "$@"; do
  d=evals/work/ld/$w; rm -rf $d; mkdir -p $d/gen/images; cp worlds/$w/boxes.json $d/; cp -R worlds/$w/shapes $d/shapes
  set -- $(.venv/bin/python -c "import json;v=json.load(open('worlds/$w/gen/views.json'));print(v['views'],'%dx%d'%tuple(v['paint']))")
  export WORLD_DIR=$d WORLD_VIEWS=$1 WORLD_PAINT=$2 WORLD_LOOKDOWN=plan
  .venv/bin/python chisel_ring.py prep || exit 1
  .venv/bin/python - <<PY
import json, shutil
v = json.load(open("worlds/$w/gen/views.json")); v["lookdown"] = json.load(open("$d/lookdown.json"))["cams"]
json.dump(v, open("$d/gen/views.json", "w"))
for i in range(v["views"]): shutil.copy("worlds/$w/gen/images/%02d.png" % i, "$d/gen/images/%02d.png" % i)
open("$d/style.txt", "w").write(v["style"])
PY
  /usr/bin/time ../spike/.venv/bin/python chisel_ring.py gen gen "$(cat $d/style.txt)" 2>&1 | grep -v "^$" | tail -12
  ls $d/gen/images | wc -l
done

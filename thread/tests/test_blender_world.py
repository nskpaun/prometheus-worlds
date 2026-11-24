"""The Blender mode: reading the model's plan and tool calls, what is sent to the Blender MCP and said back, the helper
kit with its sizing and placing of things, and the world files written at the end. No model is run; the kit's tests start Blender (no window, a few seconds) and are
skipped where Blender is not installed.
  .venv/bin/python -m unittest tests.test_blender_world        (from the thread folder)
"""
import os, sys, json, subprocess, tempfile, types, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "build"))
import chat, blender_world as W, blender_export as X

class Calls(unittest.TestCase):
    def test_a_call_is_read_with_its_code_as_written(self):
        code = 'top = box("desk_top", (1.6, 0.8, 0.04), (0, 1.2, 0.72), "oak")\nfor i in range(4):\n    print("<leg>", i)'
        text = f"Now the desk.\n\n<tool_call>\n<function=execute_blender_code>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(chat.tool_calls(text), [("execute_blender_code", {"code": code})])

    def test_several_arguments_and_no_call(self):
        text = "<tool_call>\n<function=get_scene_info>\n<parameter=query>\ndesk\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(chat.tool_calls(text), [("get_scene_info", {"query": "desk", "limit": "5"})]); self.assertEqual(chat.tool_calls("DONE"), [])

    def test_a_call_cut_short_is_still_read(self):
        self.assertEqual(chat.tool_calls("<tool_call>\n<function=execute_blender_code>\n<parameter=code>\nx = 1\ny = "), [("execute_blender_code", {"code": "x = 1\ny = "})])

    def test_shared_start_of_two_lists(self):
        self.assertEqual(chat.common([1, 2, 3, 4], [1, 2, 9]), 2); self.assertEqual(chat.common([], [1]), 0); self.assertEqual(chat.common([5, 6], [5, 6]), 2)

class Sent(unittest.TestCase):
    def tool(self, name): return types.SimpleNamespace(name=name, description="First part.\n\nSecond part.", inputSchema={"type": "object", "required": ["code", "user_prompt"],
                                 "properties": {"code": {"type": "string"}, "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "user_prompt": {"type": "string"}}})

    def test_only_the_scene_tools_are_offered_and_without_the_telemetry_field(self):
        specs = W.tool_specs([self.tool(n) for n in ("execute_blender_code", "generate_3d", "search_assets", "disable_telemetry", "get_scene_info")])
        self.assertEqual([s["function"]["name"] for s in specs], ["execute_blender_code", "get_scene_info"]); f = specs[0]["function"]
        self.assertNotIn("user_prompt", f["parameters"]["properties"]); self.assertEqual(f["parameters"]["required"], ["code"]); self.assertEqual(f["description"], "First part.")

    def test_arguments_take_the_types_the_tool_asks_for(self):
        spec = W.tool_specs([self.tool("get_scene_info")])[0]
        self.assertEqual(W.coerce({"code": "5", "limit": "5", "other": "x"}, spec), {"code": "5", "limit": 5})

    def test_a_step_runs_with_the_helpers_and_undoes_itself_on_failure(self):
        seen = []; kit = types.ModuleType("kit"); kit._begin = lambda: seen.append("begin"); kit._rollback = lambda: seen.append("rollback"); kit.box = lambda *a: seen.append(a)
        kit.__all__ = ["box"]; old = sys.modules.get("kit"); sys.modules["kit"] = kit
        try:
            exec(W.wrap('box("a", 1)\ns = """two\nlines"""'), {}); self.assertEqual(seen, ["begin", ("a", 1)])
            with self.assertRaises(ZeroDivisionError): exec(W.wrap("box(2)\n1 / 0"), {})
            self.assertEqual(seen[-1], "rollback")
        finally:
            if old is None: del sys.modules["kit"]
            else: sys.modules["kit"] = old

    def test_an_error_names_the_line_and_a_hidden_helper(self):
        t = '{"error": "x", "traceback": "Traceback (most recent call last):\\n  File \\"addon.py\\", line 9, in execute_code\\n  File \\"<step>\\", line 3, in <module>\\nTypeError: \'Object\' object is not callable\\n"}'
        out = W.explain(t, "a = 1\nlamp = group('x', [])\nlamp('y', (0, 0, 0))")
        self.assertIn("TypeError", out); self.assertIn("line 3 of your code: lamp('y', (0, 0, 0))", out); self.assertIn("(lamp)", out); self.assertNotIn("addon.py", out)

    def test_both_processes_are_cut_off_from_the_network(self):
        cmd = W.local_only(["x", "y"]); self.assertEqual(cmd[-2:], ["x", "y"])
        if os.path.exists("/usr/bin/sandbox-exec"):
            self.assertIn("(deny network-outbound)", cmd[2])
            r = subprocess.run(W.local_only(["/usr/bin/curl", "-s", "-m", "5", "-o", "/dev/null", "https://example.com"]), capture_output=True)
            self.assertNotEqual(r.returncode, 0, "a request to the internet must fail inside the sandbox")

class Plan(unittest.TestCase):
    TEXT = """Here: {"setting": "indoor", "room": {"width": 5, "depth": "4", "height": 2.7}, "doors": [{"wall": "south", "offset": 1}], "windows": [{"wall": "north"}, {"wall": "up"}],
      "wall": "#aabbcc", "floor": "red", "materials": [{"name": "Dark Oak", "color": "#553311"}],
      "things": [{"name": "mug", "what": "a mug", "size": [0.1, 0.1, 0.1], "where": {"on": "desk"}}, {"name": "door_knob", "what": "a knob"}, {"name": "Window", "what": "glass"},
                 {"name": "desk", "what": "a desk", "size": [1.4, 0.7, 0.75], "where": {"wall": "north", "offset": 0, "colour": "x"}},
                 {"name": "desk", "what": "another", "size": [1, 1], "where": {"wall": "up"}, "light": 60}, {"name": "a", "what": "a", "where": {"on": "b"}}, {"name": "b", "what": "b", "where": {"on": "a"}}]} done"""

    def test_a_plan_is_made_safe_to_build_from(self):
        p = W.read_plan(self.TEXT); names = [t["name"] for t in p["things"]]
        self.assertEqual(p["room"], [5.0, 4.0, 2.7]); self.assertEqual(p["doors"], [("south", 1.0, 0.9, 2.05)]); self.assertEqual(p["windows"], [("north", 0, 0.9, 1.4, 1.2)])
        self.assertEqual((p["wall"], p["floor"]), ("#aabbcc", "#8b6f4e")); self.assertEqual(p["materials"][0]["name"], "dark_oak"); self.assertIsNotNone(p["sky"], "a room with a window has a sky")
        self.assertNotIn("door_knob", names); self.assertNotIn("window", names); self.assertIn("desk_2", names)
        self.assertLess(names.index("desk"), names.index("mug"), "what stands on a thing is built after it")
        by = {t["name"]: t for t in p["things"]}; self.assertEqual(by["desk"]["where"], {"wall": "north", "offset": 0}); self.assertEqual(by["desk_2"]["where"], {}); self.assertEqual(by["desk_2"]["size"], [0.5, 0.5, 0.5])
        self.assertEqual(sum("on" in by[n]["where"] for n in "ab"), 1, "a ring of things standing on each other is cut")
        with self.assertRaises(ValueError): W.read_plan('{"things": []}')
        with self.assertRaises(ValueError): W.read_plan("no plan here")

    def test_outdoors_has_ground_and_sky_and_no_walls(self):
        p = W.read_plan('{"setting": "outdoor", "things": [{"name": "bench", "what": "a bench", "where": {"wall": "north", "x": 1, "y": 2}}]}')
        self.assertNotIn("room", p); self.assertEqual(p["things"][0]["where"], {"x": 1, "y": 2}); code = W.room_code(p); self.assertIn("ground(20.5,", code, "the ground reaches 8 m past the farthest thing"); self.assertIn("sky(", code); self.assertNotIn("room(", code)

    def test_the_room_is_written_from_the_plan(self):
        code = W.room_code(W.read_plan(self.TEXT)); compile(code, "room", "exec")
        self.assertIn("mat('dark_oak', '#553311'", code); self.assertIn("room(5.0, 4.0, 2.7", code); self.assertIn("doors=[('south', 1.0, 0.9, 2.05)]", code)
        ask = W.thing_ask({"name": "desk", "what": "a desk", "size": [1.4, 0.7, 0.75]}, W.read_plan(self.TEXT)); self.assertIn("1.4 wide (x) by 0.7 deep (y) by 0.75 high (z)", ask); self.assertIn("dark_oak #553311", ask)

class Rebuild(unittest.TestCase):
    """A room planned by the direct method: which of its plain things the model builds anew, and where each then stands."""
    BOXES = {"room": [5.0, 3.0, 4.0], "objects": [
        {"name": "oak desk", "look": "worn oak", "size": [1.4, 0.75, 0.7], "yaw": 3.14159265, "prims": [{"c": [2.5, 0.375, 3.65]}], "said": True},
        {"name": "wall shelf", "size": [1.0, 2.0, 0.3], "yaw": 1.57079633, "prims": [{"c": [0.15, 1.0, 1.0]}]}, {"name": "wall shelf", "size": [1.0, 2.0, 0.3], "yaw": 1.57079633, "prims": [{"c": [0.15, 1.0, 2.5]}]},
        {"name": "mug", "size": [0.1, 0.1, 0.1], "yaw": 0, "prims": [{"c": [2.5, 0.8, 3.6]}]}, {"name": "rug", "size": [2, 0.02, 3], "yaw": 0, "prims": [{"c": [2.5, 0.01, 2]}]}]}
    PLAN = {"plain": {"oak_desk": {"form": "table", "objects": [0]}, "wall_shelf": {"form": "bookcase", "objects": [1, 2]}, "mug": {"form": "jar", "objects": [3]}, "rug": {"form": "rug", "objects": [4]}, "tiny": {"form": "crate", "objects": [3]}}}
    SURF = {"parts": {"oak_desk": {"parts": {"top": {"mode": "swatch", "texture": "tex/sw_1.png", "px_per_m": 512.0, "rgb": [120, 90, 60], "cls": "wood", "material": "Worn oak, waxed"}, "leg": {"mode": "swatch", "texture": "tex/sw_1.png", "px_per_m": 512.0, "rgb": [120, 90, 60], "cls": "wood", "material": "Worn oak, waxed"},
                                            "glow": {"mode": "emit", "material": "bulb"}, "knob": {"mode": "flat", "rgb": [180, 150, 90], "cls": "metal", "material": "brass"}}}}}

    def test_furniture_is_chosen_and_small_or_flat_things_left(self):
        c = W.candidates(self.BOXES, self.PLAN, self.SURF); self.assertEqual([x["kind"] for x in c], ["oak_desk", "wall_shelf"], "the named thing first; the mug, the rug and a 10 cm crate stay as they are")
        d = c[0]; self.assertEqual(d["size"], [1.4, 0.7, 0.75], "width, depth, height in Blender's frame"); self.assertEqual(c[1]["objects"], [1, 2])
        self.assertEqual([(m["name"], m["parts"], m["texture"]) for m in d["mats"]], [("worn_oak_waxed", ["top", "leg"], "tex/sw_1.png"), ("brass", ["knob"], None)])
        ask = W.rebuild_ask(d); self.assertIn("1.4 wide (x) by 0.7 deep (y) by 0.75 high (z)", ask); self.assertIn("worn_oak_waxed: Worn oak, waxed (for its top, leg)", ask); self.assertIn("a table", ask)

    def test_a_layout_object_is_stood_in_blenders_frame(self):
        at, rot = W.spot(self.BOXES["objects"][0], self.BOXES["room"])            # against the back wall (z = depth), facing into the room
        np.testing.assert_allclose(at, [0.0, 1.65, 0.0], atol=1e-6); self.assertAlmostEqual(rot % 360, 0.0, 3)
        at, rot = W.spot(self.BOXES["objects"][1], self.BOXES["room"]); np.testing.assert_allclose(at, [-2.35, -1.0, 0.0], atol=1e-6); self.assertAlmostEqual(rot, 90.0, 3)      # on the left wall, facing +x

class Files(unittest.TestCase):
    def test_the_world_files_are_written_y_up(self):
        import trimesh
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/blender"); V = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], np.float32); T = np.array([[0, 1, 2], [0, 1, 3]], np.int32)
            np.savez(f"{d}/blender/geo.npz", CV=V, CT=T, PV=V, PT=T, PC=np.array([[0.8, 0, 0], [0, 0.4, 0]], np.float32))
            lm = np.full((16, 16, 3), 60, np.uint8); lm[5, 5] = 255; np.save(f"{d}/blender/light0.npy", lm); np.save(f"{d}/blender/env.npy", np.zeros((4, 8, 3), np.uint8))
            with open(f"{d}/scene.glb", "wb") as fh: fh.write(b"glTF")                 # Blender writes this one itself
            with open(f"{d}/blender/meta.json", "w") as fh: json.dump({"room": [5, 4, 2.7], "setting": "indoor", "lo": [0, 0, 0], "hi": [1, 2, 3], "eye": [1, 2, 1.6], "forward": [0, 1, 0], "backdrop": [0.1, 0.2, 0.3], "things": 3, "lightmaps": 1, "light_max": 8.0}, fh)
            X.main(d)
            with open(f"{d}/scene.json") as fh: s = json.load(fh)
            self.assertEqual((s["up"], s["floor"], s["metre"], s["room"]), ([0, 1, 0], 0.0, 1.0, [5, 2.7, 4])); self.assertEqual(s["eye"], [1, 1.6, -2]); self.assertEqual(s["forward"], [0, 0, -1])
            self.assertEqual((s["model"], s["lightmaps"], s["env"], s["light_max"]), ("scene.glb", ["light0.png"], "env.jpg", 8.0))
            got = np.asarray(Image.open(f"{d}/light0.png")); self.assertEqual(got.shape, (16, 16, 3)); self.assertLess(int(got.max()), 80, "one bright speck of the path tracer is taken out")
            f = trimesh.load(f"{d}/mesh.glb", force="mesh"); self.assertEqual(len(f.faces), 2); self.assertEqual(tuple(f.visual.vertex_colors[0][:3]), (204, 0, 0)); np.testing.assert_allclose(f.bounds, [[0, 0, -2], [1, 3, 0]], atol=1e-6)
            self.assertEqual(len(trimesh.load(f"{d}/collider.glb", force="mesh").faces), 2)

class Windows(unittest.TestCase):
    def test_the_layouts_windows_are_handed_to_the_room_builder(self):
        sys.path.insert(0, os.path.join(HERE, "..", "build")); import level
        b = {"room": [5.0, 3.0, 6.0], "openings": [{"kind": "window", "wall": "left", "along": 3.0, "width": 2.0, "height": 1.8, "sill": 0.6}, {"kind": "window", "wall": "front", "along": 1.0, "width": 1.0, "height": 1.0},
                                                    {"kind": "door", "wall": "back", "along": 2.0, "width": 0.9, "height": 2.1}, {"kind": "window", "wall": "roof", "along": 1, "width": 1, "height": 1}]}
        self.assertEqual(level.opening_hints(b), [{"kind": "window", "wall": "x0", "u0": 2.0, "u1": 4.0, "v0": 0.6, "v1": 2.4}, {"kind": "window", "wall": "z0", "u0": 3.5, "u1": 4.5, "v0": 0.9, "v1": 1.9}])
        self.assertEqual(level.opening_hints({"room": [4, 3, 4]}), [])

KIT = r'''
import sys, json, importlib.util, bpy
spec = importlib.util.spec_from_file_location("kit", sys.argv[sys.argv.index("--") + 1]); m = importlib.util.module_from_spec(spec); sys.modules["kit"] = m; spec.loader.exec_module(m)
from kit import *
import kit
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
out = {"empty": check()}
room(5, 4, 2.7, doors=[("south", 1.5, 0.9, 2.05)], windows=[("north", 0, 0.9, 1.6, 1.2)])
names = [o.name for o in bpy.data.objects]; out["shell"] = {"door": any(n.startswith("door_south") for n in names), "pane": sum(bool(o.get("collide_only")) for o in bpy.data.objects), "room": list(bpy.context.scene["room"])}
oak = mat("oak", "#8a5a2b", 0.5); top = box("top", (1.6, 0.8, 0.04), (0, 1.2, 0.72), oak, bevel=0.005)
desk = group("desk", [top] + [cyl("leg", 0.03, 0.72, (x, y, 0), oak) for x in (-0.7, 0.7) for y in (0.9, 1.5)])
lathe("vase", [(0.05, 0), (0.09, 0.1), (0.04, 0.25)], (0.3, 1.2, 0.76), "blue"); sphere("ball", 0.2, (1, -1, 1.0), "red"); box("crate", (1, 1, 1), (3, 0, 0), oak)
extrude("counter", [(0, 0), (1, 0), (1, 0.5), (0, 0.5)], 0.9, (-2.4, -1.9, 0), "stone"); tube("rail", [(-2, 1.93, 1), (-1, 1.93, 1)], 0.01, "steel")
out["things"] = things().splitlines(); out["wrong"] = check().splitlines()
place("ball", (1, -1, 0)); place("crate", (1.8, -1.2, 0), 30); lamp("bulb", (0, 0, 2.4), 100); out["mended"] = check().splitlines()
kit._begin(); box("extra", (1, 1, 1)); bpy.context.scene["room"] = [9, 9, 9]; kit._rollback(); out["undone"] = ["extra" not in bpy.data.objects, list(bpy.context.scene["room"])]
place("desk", rot=90); lo, hi = kit._bounds([o for o in kit._family(bpy.data.objects["desk"]) if o.type == "MESH"]); out["turned"] = [round(hi.x - lo.x, 2), round(hi.y - lo.y, 2)]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
room(5, 4, 2.7); size = lambda n: [round(v, 2) for v in kit._size(bpy.data.objects[n])]; stands = lambda n: [round(v, 2) for v in kit._bounds([o for o in kit._family(bpy.data.objects[n]) if o.type == "MESH"])[0]]
def thing(name, build): kit._begin(); build(); return kit._collect(name)
thing("case", lambda: [box("side", (0.03, 0.3, 3.4), (sx * 0.485, 0, 0)) for sx in (-1, 1)] + [box("shelf", (0.94, 0.28, 0.03), (0, 0, 0.1 + i * 0.6)) for i in range(5)])
out["fit"] = [kit._fit("case", [1.0, 0.3, 2.0]), size("case")]; kit._put("case", {"wall": "west", "offset": 0.5}); out["case"] = [stands("case"), size("case")]
thing("desk", lambda: box("top", (1.4, 0.7, 0.75), (3, 3, 1))); kit._put("desk", {"wall": "north", "offset": 0}); out["desk"] = stands("desk")
thing("chest", lambda: box("c", (1.0, 0.5, 0.5))); kit._put("chest", {"wall": "north", "offset": 0.2}); out["chest"] = stands("chest")
thing("mug", lambda: cyl("m", 0.04, 0.1)); out["mug_says"] = kit._put("mug", {"on": "desk", "dx": 5}); out["mug"] = stands("mug")
thing("pot", lambda: cyl("p", 0.1, 0.9)); out["pot_says"] = kit._put("pot", {"on": "case"}); out["pot"] = [stands("pot")[2], size("pot")[2]]
thing("stool", lambda: cyl("s", 0.2, 0.45)); kit._put("stool", {"x": 0, "y": 1.6, "rot": 0}); out["stool"] = stands("stool")
thing("art", lambda: box("a", (0.6, 0.03, 0.4))); kit._put("art", {"hang": "east", "offset": 0, "height": 1.4}); out["art"] = [stands("art"), size("art")]
thing("fan", lambda: cyl("f", 0.3, 0.2)); kit._put("fan", {"ceiling": True, "x": 0, "y": 0}); out["fan"] = stands("fan")[2]; kit._glow("fan", 80); out["glow"] = round(bpy.data.objects["fan_light"].location.z, 2)
out["clean"] = check()
out["exact"] = kit._exact("chest", [0.8, 0.4, 0.6]); two = kit._copy("desk", "desk_b"); place(two, (0, -1.2, 0), 90); out["copy"] = [two, size(two), size("desk"), len(kit._family(bpy.data.objects[two]))]
bpy.data.objects[two]["thing"] = True; bpy.data.objects[two]["object"] = 7; kit._drop_object(7); out["dropped"] = two not in bpy.data.objects
m = kit._texmat("plain_oak", "", 512, [120, 90, 60], "wood"); out["texmat"] = [m.name, round(m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value, 2)]
import os, tempfile
snap = os.path.join(tempfile.mkdtemp(), "look.png"); kit._snap("case", snap, 64); out["snap"] = [os.path.getsize(snap) > 200, bpy.context.scene.camera is None, all(not o.hide_render for o in bpy.data.objects), "snap_cam" in bpy.data.objects]
print("RESULT " + json.dumps(out))
'''

class Kit(unittest.TestCase):
    def test_the_helpers_build_and_the_check_finds_what_is_wrong(self):
        try: app = W.find_blender()
        except SystemExit: self.skipTest("Blender is not installed")
        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/k.py", "w") as fh: fh.write(KIT)
            r = subprocess.run([app, "-b", "--factory-startup", "--python", f"{d}/k.py", "--", os.path.join(HERE, "..", "blender_kit.py")], capture_output=True, text=True, timeout=120)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None); self.assertIsNotNone(line, r.stdout[-800:] + r.stderr[-800:]); o = json.loads(line[7:])
        self.assertIn("no room", o["empty"]); self.assertEqual(o["shell"], {"door": True, "pane": 1, "room": [5.0, 4.0, 2.7]})
        self.assertIn("desk: 1.60 x 0.80 x 0.76 m, stands at (0.00, 1.20, 0.00)", o["things"]); wrong = "\n".join(o["wrong"])
        for word in ("no light", "ball hangs in the air", "crate sticks 1.00 m out through a wall"): self.assertIn(word, wrong)
        self.assertNotIn("vase", wrong, "the vase stands on the desk"); self.assertNotIn("rail", wrong, "the rail is on the wall")
        self.assertEqual(o["mended"], []); self.assertEqual(o["undone"], [True, [5.0, 4.0, 2.7]]); self.assertEqual(o["turned"], [0.8, 1.6])
        self.assertEqual(o["fit"], [["z"], [1.0, 0.3, 2.0]], "a thing far from its planned size is brought to it")
        self.assertEqual(o["case"], [[-2.5, 0.0, 0.0], [0.3, 1.0, 2.0]], "against the west wall, turned to face the room")
        self.assertEqual(o["desk"], [-0.7, 1.3, 0.0]); self.assertGreaterEqual(o["chest"][0], 0.7 - 0.01, "a taken spot is left for the nearest free one along the wall")
        self.assertIn("stands on desk", o["mug_says"]); self.assertEqual(o["mug"][2], 0.75); self.assertLessEqual(o["mug"][0], 0.7)
        self.assertIn("shelf", o["pot_says"]); self.assertLess(o["pot"][0], 2.0); self.assertLessEqual(o["pot"][0] + o["pot"][1], 2.7, "on a shelf with room for it, not on top through the ceiling")
        self.assertLess(o["stool"][1] + 0.4, 1.3 + 0.01, "kept clear of the desk"); self.assertEqual(o["art"], [[2.47, -0.3, 1.4], [0.03, 0.6, 0.4]])
        self.assertEqual(o["fan"], 2.5); self.assertEqual(o["glow"], 2.44); self.assertEqual(o["clean"], "")
        self.assertEqual(o["exact"], [0.8, 0.4, 0.6]); self.assertEqual(o["copy"], ["desk_b", [0.7, 1.4, 0.75], [1.4, 0.7, 0.75], 2], "a copy, parts and all, turned on its own"); self.assertTrue(o["dropped"])
        self.assertEqual(o["texmat"], ["plain_oak", 0.5]); self.assertEqual(o["snap"], [True, True, True, False], "a picture is taken and the scene left as it was")

if __name__ == "__main__":
    unittest.main()

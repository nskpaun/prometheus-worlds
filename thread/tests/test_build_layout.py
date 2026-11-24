"""The new method's layout (build/layout.py): its repairs, on lists of things written badly on purpose, and the rules
that fill a room. Each test places a list as the world-plan model would have written it; no model is run."""
import os, sys, unittest
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]
import layout as L

def W(things, room=(5, 3, 6)): return L.place("", {"room": {"width": room[0], "height": room[1], "depth": room[2]}, "things": things})
def get(w, name): return next(o for o in w["objects"] if o["name"] == name)
top = lambda o: o["prims"][0]["c"][1] + (o["prims"][0]["h"][1] if o["prims"][0]["shape"] == "box" else o["prims"][0]["hy"])
bot = lambda o: 2 * o["prims"][0]["c"][1] - top(o)
desk = {"name": "desk", "said": True, "form": "sculpted", "size": [1.5, 0.76, 0.75], "place": {"floor": "free", "at": [0.5, 0.5]}}

class Repairs(unittest.TestCase):
    def ok(self, cond, what): self.assertTrue(cond, what)
    def test_01_a_globe_1_2_m_wide_and_1_8_m_high_on_the_desk(self):
        ok = self.ok
        w = W([desk, {"name": "globe", "said": True, "form": "sculpted", "size": [1.2, 1.8, 1.2], "place": {"on": "desk"}}]); g = get(w, "globe")
        ok(g["size"][0] <= 0.75 and abs(bot(g) - 0.76) < 1e-6 and g["on"] == 0, f"an absurd globe on the desk is shrunk to fit its top and rests on it: {g['size']}")
    def test_02_a_car_on_a_platform_too_small_for_it_the_platform_grows(self):
        ok = self.ok
        w = W([{"name": "platform", "said": True, "form": "crate", "size": [3, 0.4, 1.2], "place": {"floor": "free", "at": [0.5, 0.5]}}, {"name": "car", "said": True, "form": "sculpted", "size": [1.8, 1.5, 4.4], "place": {"on": "platform"}}], (14, 4.5, 12))
        ok(get(w, "platform")["size"][2] >= 4.4 and get(w, "car")["size"] == [1.8, 1.5, 4.4] and abs(bot(get(w, "car")) - 0.4) < 1e-6, "a platform too small for the car is grown, the car keeps its size and stands on it")
    def test_02b_a_car_on_a_tall_sculpted_platform_shorter_than_the_car(self):
        ok = self.ok
        w = W([{"name": "display platform", "said": True, "form": "sculpted", "size": [2.5, 1.5, 2.5], "place": {"floor": "free", "at": [0.5, 0.5]}}, {"name": "car", "said": True, "form": "sculpted", "size": [1.9, 1.4, 4.6], "place": {"on": "display platform"}}], (14, 4.5, 12))
        p, c = get(w, "display platform"), get(w, "car")
        ok(c["size"] == [1.9, 1.4, 4.6] and p["form"] == "crate" and p["size"][1] <= 0.6 and max(p["size"][0], p["size"][2]) >= 4.6 and abs(bot(c) - p["size"][1]) < 1e-6, f"the car keeps its size on a plain low platform that holds it: {p['size']} {c['size']}")
    def test_03_a_fireplace_written_as_fixed_to_the_wall_with_its_foot_on_th(self):
        ok = self.ok
        w = W([{"name": "fireplace", "said": True, "form": "sculpted", "size": [1.2, 1.1, 0.6], "place": {"wall": "right", "along": 0.5, "height": 0.55}}]); f = get(w, "fireplace")
        ok(f["attach"] == "floor" and bot(f) == 0 and abs(f["prims"][0]["c"][0] - (5 - 0.3)) < 1e-6, "a fireplace 'on the wall' stands on the floor against that wall")
    def test_04_three_things_written_at_the_same_spot_do_not_intersect(self):
        ok = self.ok
        w = W([dict(desk, name=f"desk {k}", said=False) for k in range(3)]); ok(not w["unsolved"] and len(w["objects"]) == 3, "three desks written at one spot are moved apart")
    def test_05_more_bookcases_than_the_wall_is_long_the_rest_go_to_other_wa(self):
        ok = self.ok
        w = W([{"name": "bookcase", "said": True, "form": "sculpted", "size": [1.2, 2.2, 0.35], "count": 8, "place": {"floor": "back", "along": 0.5}}]); ok(not w["unsolved"] and len(w["objects"]) == 8, f"eight bookcases on a 5 m wall: all placed on walls with room, none intersecting")
    def test_06_a_lantern_hung_down_to_head_height_over_open_floor_and_one_o(self):
        ok = self.ok
        w = W([desk, {"name": "lantern", "said": True, "form": "sculpted", "size": [0.3, 0.45, 0.3], "place": {"ceiling": [0.5, 0.5], "drop": 2.2}}, {"name": "lamp", "form": "sculpted", "size": [0.3, 0.45, 0.3], "place": {"ceiling": [0.15, 0.15], "drop": 2.0}}])
        ok(bot(get(w, "lantern")) >= 0.76 + 0.3 - 1e-6 and bot(get(w, "lamp")) >= 2.0 - 1e-6, f"hanging things clear the desk under them ({bot(get(w, 'lantern')):.2f} m) and heads over open floor ({bot(get(w, 'lamp')):.2f} m)")
    def test_07_a_picture_written_thick_a_rug_written_tall_a_curtain_written(self):
        ok = self.ok
        w = W([{"name": "picture", "form": "frame", "size": [0.04, 0.6, 0.8], "place": {"wall": "left", "along": 0.5, "height": 1.5}}, {"name": "rug", "form": "rug", "size": [2.4, 1.6, 0.02], "place": {"floor": "free", "at": [0.5, 0.5]}}])
        ok(get(w, "picture")["size"][2] <= 0.08 and get(w, "rug")["size"][1] == 0.02, "flat things are turned flat: picture depth " + str(get(w, "picture")["size"][2]))
    def test_08_a_thing_on_a_thing_that_is_not_in_the_list_a_relation_loop(self):
        ok = self.ok
        w = W([{"name": "vase", "form": "pot", "size": [0.2, 0.3, 0.2], "place": {"on": "mantelpiece"}}, {"name": "a", "form": "crate", "size": [0.5, 0.5, 0.5], "place": {"on": "b"}}, {"name": "b", "form": "crate", "size": [0.5, 0.5, 0.5], "place": {"on": "a"}}])
        ok(not w["unsolved"] and get(w, "vase")["attach"] == "floor", "a thing on something that does not exist stands on the floor; a loop of 'on' is cut")
    def test_09_things_added_until_the_floor_is_cut_in_two_are_dropped_named(self):
        ok = self.ok
        w = W([dict(desk, said=True)] + [{"name": f"crate {k}", "form": "crate", "size": [1.0, 1.0, 1.0], "place": {"floor": "free", "at": [0.1 + 0.2 * (k % 5), 0.3 + 0.2 * (k // 5)]}} for k in range(15)])
        ok(any(o["name"] == "desk" for o in w["objects"]) and w["dropped"] and not w["unsolved"], f"a floor filled with crates: {len(w['dropped'])} added things dropped, the named desk kept, walkable again")
    def test_10_shelf_with_books_deeper_than_it_a_wall_thing_over_a_door_not(self):
        ok = self.ok
        w = W([{"name": "door", "form": "door", "size": [0.9, 2.1, 0.05], "place": {"floor": "back", "along": 0.5}}, {"name": "cabinet", "form": "sculpted", "size": [1.2, 1.8, 0.5], "place": {"floor": "back", "along": 0.5}},
               {"name": "clock", "form": "sculpted", "size": [0.4, 0.4, 0.1], "place": {"wall": "back", "along": 0.5, "height": 1.5}}])
        c = get(w, "cabinet")["prims"][0]["c"][0]; k = get(w, "clock")["prims"][0]["c"]
        ok(abs(c - 2.5) >= 0.45 + 0.6 and not (abs(k[0] - 2.5) < 0.65 and k[1] < 2.3) and len(w["openings"]) == 1, "a cabinet and a clock written across the door are moved off it")
    def test_11_things_on_a_small_table_packed_without_overlap_extra_ones_dr(self):
        ok = self.ok
        w = W([dict(desk, name="side table", size=[0.5, 0.6, 0.5])] + [{"name": f"jar {k}", "form": "jar", "size": [0.15, 0.2, 0.15], "place": {"on": "side table"}} for k in range(12)])
        ok(not w["unsolved"] and 3 <= sum(o["attach"] == "on" for o in w["objects"]) < 12, f"twelve jars on a small table: {sum(o['attach'] == 'on' for o in w['objects'])} fit, the rest are dropped, none overhangs")
    def test_12_the_sense_check_shared_shapes_and_the_filling_rules(self):
        ok = self.ok
        car = lambda k, nm: {"n": k, "name": nm, "said": True, "form": "sculpted", "size": [1.8, 1.4, 4.4], "place": {"floor": "corner", "corner": ["back left", "back right", "front left", "front right", "back left"][k - 1]}}
        raw = {"room": {"width": 14, "height": 4.5, "depth": 12}, "things": [car(1, "muscle car"), car(2, "open tourer"), car(3, "saloon car"), car(4, "sports coupe"), car(5, "town car"),
               {"n": 6, "name": "fireplace", "form": "sculpted", "size": [1.2, 1.2, 0.5], "place": {"floor": "back", "along": 0.5}}, {"n": 7, "name": "mantel clock", "form": "sculpted", "size": [0.3, 0.3, 0.1], "place": {"on": "fireplace"}},
               {"n": 8, "name": "wool rug", "form": "rug", "size": [8.5, 0.02, 6], "place": {"floor": "free", "at": [0.5, 0.5]}}, {"n": 9, "name": "wooden stool", "form": "sculpted", "size": [0.4, 0.45, 0.4], "place": {"floor": "free", "at": [0.3, 0.5]}},
               {"n": 10, "name": "tool bench", "form": "table", "size": [1.6, 0.9, 0.7], "place": {"floor": "back", "along": 0.2}}, {"n": 11, "name": "concrete floor", "form": "rug", "size": [14, 0.02, 12], "place": {"floor": "free", "at": [0.5, 0.5]}}]}
        chk = {"drop": [6], "fullness": 2, "props": [{"name": "oil can", "look": "red tin", "form": "bottle", "size": [0.1, 0.25, 0.1]}, {"name": "spark plugs", "look": "boxed", "form": "crate", "size": [0.15, 0.08, 0.1]}],
               "storage": [{"name": "tyre stack", "look": "black rubber", "form": "rings", "size": [0.7, 0.9, 0.7]}, {"name": "parts shelves", "look": "grey steel", "form": "bookcase", "size": [1.2, 1.8, 0.5]}], "walls": [{"name": "enamel sign", "look": "old petrol brand", "form": "frame", "size": [0.8, 0.8, 0.03]}]}
        w = L.place("A modern garage with 5 antique cars on display.", raw, chk); names = [o["name"] for o in w["objects"]]
        ok("fireplace" not in names and "mantel clock" not in names, "the sense check drops the fireplace from the garage, and the mantel clock goes with it")
        ok(len({o["kind"] for o in w["objects"] if o["said"]}) == L.VARIANT_SHAPES and sum(o["said"] for o in w["objects"]) == 5, f"five different cars share {L.VARIANT_SHAPES} shapes, each keeping its own name and look")
        ok(get(w, "wooden stool")["form"] == "table" and "concrete floor" not in names, "a leggy stool is a plain form; 'concrete floor' is the room, not a thing")
        r = get(w, "wool rug")["size"]; ok(r[0] * r[2] <= L.RUG_M2 + 1e-2, f"an added 8.5 x 6 m rug is cut to {r[0]:.1f} x {r[2]:.1f} m ({L.RUG_M2:g} m2 or {L.RUG_SHARE:.0%} of the floor at most)")
        c = w["counts"]; ok(c["filled_by_rule"] >= 8 and any(o.get("fill") == "wall" for o in w["objects"]) and any(o.get("fill") == "top" and w["objects"][o["on"]]["name"] == "tool bench" for o in w["objects"]) and not w["unsolved"],
           f"the filling rules add {c['filled_by_rule']} things: on the bench, along the walls, on the walls; nothing intersects")
        small = L.place("A study.", {"room": {"width": 5, "height": 3, "depth": 5}, "things": [dict(desk, n=1)]}, chk)["counts"]["filled_by_rule"]
        big = L.place("A study.", {"room": {"width": 12, "height": 3, "depth": 12}, "things": [dict(desk, n=1)]}, chk)["counts"]["filled_by_rule"]
        ok(big >= 1.4 * small, f"the same list (and a filler pool of three things) in a room with 2.4 times the walls gets more filling: {small} things at 5 x 5 m, {big} at 12 x 12 m")
        w = W([{"name": "heavy curtains", "said": True, "form": "cloth", "size": [1.4, 2.4, 0.15], "place": {"wall": "left", "along": 0.3, "height": 1.5}}, {"name": "window", "form": "window", "size": [1.0, 1.4, 0.1], "place": {"wall": "left", "along": 0.45, "height": 1.6}}])
        cu = get(w, "heavy curtains"); ok(cu["size"][1] >= 2.8 and bot(cu) <= 0.05 and cu["size"][0] >= 2.0 and cu.get("opts", {}).get("gap", 0) > 0.3 and abs(cu["prims"][0]["c"][2] - w["openings"][0]["along"]) < 0.05,
           f"heavy curtains reach from floor to ceiling ({cu['size'][1]:.2f} m of 3 m), are centred on the window near them and drawn apart to show it")

if __name__ == "__main__": unittest.main()

class Seating(unittest.TestCase):
    """Chairs written as rows of free chairs are set round the tables, facing them; coffee tables are left alone."""
    def raw(self, chairs=10, tables=2, kind="round dining table"):
        th = [{"name": kind, "said": True, "form": "table", "size": [0.8, 0.75, 0.8], "place": {"floor": "free", "at": [0.45 + 0.05 * k, 0.5]}} for k in range(tables)]
        th += [{"name": "wooden chair", "said": True, "form": "sculpted", "size": [0.45, 0.85, 0.45], "place": {"floor": "free", "at": [0.2 + 0.06 * k, 0.3]}} for k in range(chairs)]
        return {"room": {"width": 6, "depth": 8, "height": 3}, "things": th}

    def test_chairs_round_tables_facing_them(self):
        import math
        w = L.place("A dining room: two round dining tables with chairs in the middle.", self.raw())
        tabs = [o for o in w["objects"] if "table" in o["name"]]; chs = [o for o in w["objects"] if "chair" in o["name"]]
        self.assertEqual(len(tabs), 2); self.assertEqual(len(chs), 8)                     # four a table, two left out
        self.assertGreater(abs(tabs[0]["prims"][0]["c"][0] - tabs[1]["prims"][0]["c"][0]) + abs(tabs[0]["prims"][0]["c"][2] - tabs[1]["prims"][0]["c"][2]), 1.2)   # spread, not piled
        for c in chs:
            x, z = c["prims"][0]["c"][0], c["prims"][0]["c"][2]; t = min(tabs, key=lambda t: (t["prims"][0]["c"][0] - x) ** 2 + (t["prims"][0]["c"][2] - z) ** 2)
            dx, dz = t["prims"][0]["c"][0] - x, t["prims"][0]["c"][2] - z
            self.assertLess(math.hypot(dx, dz), 1.3)
            self.assertGreater(math.sin(c["yaw"]) * dx + math.cos(c["yaw"]) * dz, 0)        # its front is toward its table

    def test_coffee_tables_are_not_sat_at(self):
        w = L.place("A lounge.", self.raw(chairs=3, tables=1, kind="coffee table"))
        self.assertFalse(any("chairs set round" in r for r in w["repairs"]))

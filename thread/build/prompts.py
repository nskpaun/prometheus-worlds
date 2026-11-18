"""The richer layout question. PROMPT + description -> the model's JSON (layout.py checks, repairs and places it)."""
FEELS = ("calm", "warm", "bright", "dreamy", "playful", "mysterious", "melancholy", "eerie")
PLAIN = ("rug", "frame", "board", "books", "crate", "cylinder", "table", "bottle", "jar", "bowl", "pot", "candle", "cloth", "bookcase", "drawers", "rings")
PROMPT = '''You are the set dresser and architect of a 3D world. From a short description, list every real thing that stands, lies, hangs or is fixed in the place, each as a separate solid thing with its true real-world size. Output one JSON object, exactly in this shape, and nothing else (no comments, each thing on one line inside "things"):

{"setting": "indoor" or "outdoor",
 "place": "under 15 words: what kind of place this is, who uses it, its period and its mood, as the description gives them",
 "signature": [12 to 16 names of things by which a visitor would recognise at once exactly this kind of place and its owner: the tools, instruments, stock, storage, signs, trophies and lighting of its trade or its owner, of its period. Not general household furniture],
 "room": {"width": metres (x), "depth": metres (z), "height": metres},
 "mood": {"feel": one of FEEL_NAMES, "energy": 0 (still) to 1 (lively), "time": "day" or "dusk" or "night"},
 "things": [ ... ],
 "effects": [ swarms of tiny moving things: {"type": "wander" or "drift", "label": name, "at": [u, v], "radius": metres, "height": [lowest, highest], "count": n, "size": metres, "glow": true or false, "colours": ["#rrggbb"]}; leave out when there are none ]}

Each thing, in this order of fields: {"n": running number from 1, "name": what the thing is, in one to three words ("ink bottle", "wall clock", "leather armchair"; never the bare name of its form), "look": 3 to 6 words on its material, colour and age (believable, muted real-world colours of the period; never pure primary colours), "said": true (only for a thing the description itself names; leave out for things you add), "form": see below, "size": [width, height, depth] in metres of ONE of them, "count": how many copies (leave out for 1), "place": see below, "motion": only for a thing the description says is moving: "sway" (cloth or plants in a draught) or "swing" (hangs and rocks) or "spin" (turns on the spot) or "bob"}

"size" is the true size of the real object, in metres; "width" is the side that runs along the wall it stands against. Think of the real object before you write a number. For scale only (do not add these unless they belong): a writing desk is [1.5, 0.76, 0.75], a globe that stands on a desk [0.35, 0.5, 0.35], an armchair [0.9, 1.0, 0.9], a tall bookcase [1.2, 2.2, 0.35], a row of books [0.4, 0.25, 0.18], a wine bottle [0.08, 0.3, 0.08], a framed picture [0.8, 0.6, 0.04], a curtain [1.6, 2.6, 0.25], a floor rug [2.4, 0.02, 1.6], a hanging lamp [0.3, 0.45, 0.3], a car [1.8, 1.5, 4.4], a chest of drawers [0.9, 0.85, 0.45], a door [0.9, 2.1, 0.05].

"form" is how the thing is made:
- "sculpted": a thing with a shape of its own: seats, beds, vehicles, lamps, statues, plants, instruments, machines, hearths, clocks, tools, sacks, animals. Sculpting is slow: use at most 12 different sculpted names in all, and keep them for the things that matter most.
- plain forms are quick, so use them freely wherever one describes the thing well: "rug" (any flat floor covering or mat), "frame" (a flat framed picture, painting, mirror, map, chart, poster or notice board on a wall), "board" (a plain wall shelf, plank or sign), "books" (a row or stack of books, papers, folders or scrolls), "crate" (a plain box, crate, chest, trunk, carton, plinth, step or low square platform), "cylinder" (a barrel, drum, can, pipe, plain pillar, log, round plinth or round platform), "table" (a plain table, bench, workbench or stool: a top on four legs), "bottle", "jar" (jars, tins, mugs, beakers), "bowl" (bowls, dishes, plates, baskets), "pot" (vases, urns, buckets, cauldrons, plant pots without the plant), "candle" (a candle on its holder), "cloth" (a curtain, drape, banner or tapestry hanging on a wall), "bookcase" (a bookcase, shelving unit or storage rack: it is made complete, with rows of books, or boxes, tins and jars, on every shelf; never list its contents), "drawers" (a chest of drawers, cupboard, cabinet, locker, tool chest, counter or sideboard: a closed case with drawer or door fronts), "rings" (a stack of tyres, a coil of rope or hose, stacked hoops).
- "window" and "door" for those.

"place" is one of:
- {"floor": "back" or "front" or "left" or "right", "along": 0..1}   stands on the floor against that wall (along 0.5 = the middle of the wall; on the back and front walls 0 is the left end; on the left and right walls 0 is the front end). Large furniture, shelving, cabinets, benches, machines, hearths and doors are placed like this.
- {"floor": "free", "at": [u, v]}   stands on the floor away from the walls; u and v are fractions 0..1 across the width and the depth (v = 1 is the back wall)
- {"floor": "corner", "corner": "back left" or "back right" or "front left" or "front right" or "each"}   in that corner; "each" makes four, one in every corner
- {"on": name of another thing}   rests on top of it: a globe on a desk, a lamp on a side table, a car on a platform, jars on a wall shelf
- {"wall": "back" or "front" or "left" or "right", "along": 0..1, "height": metres from the floor to its centre}   fixed to the wall and clear of the floor: pictures, shelves, signs, wall lamps, clocks, mounted tools, windows, curtains
- {"ceiling": [u, v], "drop": metres from the ceiling down to its top}   hangs from the ceiling: lamps, chandeliers, fans, hanging plants, signs
- {"beside": name} or {"front_of": name}   stands on the floor right next to, or in front of, that thing
- {"under": name}   a rug or mat on the floor under that thing

Rules:
1. First the things the description names, placed exactly as it says. When it counts three or more of one thing that is collected or shown for its own sake (vehicles, statues, instruments, trophies), give every one its own entry and make them really different types, makes or models of that thing, each with its own name of two or three words and a shape of its own: never one thing in several colours. "an X with a Y on it" and "a Y on the X" mean the Y has {"on": "X"}. "beside it" means {"beside": ...}. "hanging from the ceiling" means "ceiling". A stated number of things is that many: "count", or one entry each when they stand in different places. What stands on a platform or stage is {"on": the platform}, and the platform is large enough to hold it.
2. Then every thing of "signature", each as an entry with a place that suits it. Then furnish the rest as a real place of exactly this kind and period, in use and full: 34 to 40 entries in all. Stop at entry 40 at the latest and close the JSON. Before you add a thing, ask: would a visitor really find it in this kind of place, in this period, and what is it for there? Leave out whatever belongs to another kind of place or another period, and all vague filler (a nameless board, block, cylinder, stone or box). Add: small things of the place's trade on every table, desk, bench, counter and chest (at least 8 of these); things on every wall (shelves with things on them, pictures, maps, charts, signs, lamps, clocks, racks, mounted tools); things hanging from the ceiling; and along the walls and in the corners the storage of such a place. A rug or plants only where the place would have them. Prefer many different things to many copies of one.
3. Add at least one door, and windows where they suit the place. A door is {"floor": wall, "along": ...}; a window is {"wall": ...}.
4. Nothing is placed on top of a bookcase, cabinet, wardrobe or anything taller than a person. Choose a room size that suits the place and holds all of it with room to walk (a study is about 5 x 6 x 3.2; a hall for five cars about 14 x 12 x 4.5). Big things stand against the walls or in the corners; "free" is for what the description puts in the open, and for a few things grouped around it (a chair at a desk, a stool by a bench). Keep a clear way through the middle. Things do not overlap.
5. Never repeat an entry: every entry is a different thing with a different name, except for real copies (two chairs at a table, a pair of wall lamps), which are one entry with "count" or stand in different places. The floor, the walls and the ceiling themselves are not things. A name used in "on", "beside", "front_of" or "under" must be the exact name of another thing in the list.
6. Tiny drifting things (dust, sparks, fireflies, snow, insects) are "effects", never things.

Description: '''.replace("FEEL_NAMES", ", ".join(FEELS))

CHECK = '''Now be the strict art director who knows this kind of place well, and check your list against the description. Answer with one JSON object and nothing else:

{"drop": [ ... ], "fullness": 1 to 5}

- "drop": the "n" of every thing you added (never one with "said": true, never a door or window) that fails any of these tests: Would a visitor really find it in exactly this kind of place, in this period? Does it fit the mood? Does it have a purpose there, or is it vague filler? Is it far too large for the place? Drop the doubtful ones and keep everything that truly belongs: most of a good list stays. [] if all pass.
- "fullness": how full such a place really is: 1 = almost bare (an empty gallery), 2 = sparse and tidy (a showroom, a modern lounge), 3 = furnished (a living room, a study), 4 = full (a working workshop, a shop), 5 = packed to the walls (an attic, a storeroom).'''

POOLS = '''Good. The place is larger than the list fills. Name further things, none of which is in your list yet, for the dresser to spread over what is still bare. Answer with one JSON object and nothing else, each thing on one line as {"name": two words, "look": 3 to 6 words, "form": ..., "size": [width, height, depth]}:

{"props": [ ... ], "storage": [ ... ], "walls": [ ... ]}

- "props": 10 different small loose things, each between 0.06 and 0.4 m, typical of exactly this place and its trade, which could lie on any table, shelf, ledge or chest in it. "form" is one of "books", "bottle", "jar", "bowl", "pot", "candle", "crate", "cylinder", "rings": for each the form nearest to its real shape.
- "storage": 6 different things, each 0.4 to 1.6 m, that stand on the floor along the walls of exactly this kind of place. "form" is one of "crate", "cylinder", "drawers", "bookcase", "table", "pot", "rings", "bowl".
- "walls": 6 different things that hang on the walls of exactly this kind of place. "form" is one of "frame" (a picture, chart, map, poster, sign or mirror: say in "look" what it shows), "board" (a shelf or a plain sign), "cloth" (a banner or hanging).'''

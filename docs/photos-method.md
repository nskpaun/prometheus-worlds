# Photos to world: pieces, not training

How `pipeline.py photos` builds a world from photos. Paths are relative to the repository root.

## Why not train a splat

The earlier method solved where each photo was taken and then trained a Gaussian splat until its renders matched the photos. On a real room that failed in a specific way. A bathroom shot in 67 photos came out full of needles, haze and smeared walls, and training four times longer made it worse (18.3 dB on held-out photos fell to 17.8 dB). The causes were in what the trainer was given:

- Blank walls, a white curtain and a ceiling have no texture, so nothing told the trainer where those surfaces were.
- The photos differed in brightness by about a quarter and in colour by about a tenth, and the trainer invented haze to explain it.
- Ten photos could not be placed at all, and ten more were held by fewer than 150 matched points.

## The method

SHARP (Apple, 2025) turns one photo into about 1.2 million splats in a single pass, with no training. Seen from near where the photo was taken, that piece is clean: flat walls, sharp edges, no needles. A room is then the pieces of all its photos, put where their photos were taken and merged.

1. **Match** (`thread/photo_match.py`). ALIKED keypoints and LightGlue matching for every pair of photos. They find matches on plain walls and repeating tile where SIFT finds too little: on the bathroom, 62 of 67 photos placed against 57, with three times the points per photo.
2. **Solve** (`thread/sfm.py`). COLMAP checks the matches for geometry and solves the cameras. If the matcher is not installed, COLMAP's own SIFT is used.
3. **Pieces** (`thread/photo_pieces.py`). SHARP on every placed photo, with the solved focal length. About 6 seconds a photo.
4. **Merge** (`thread/photo_fuse.py`), in seven steps:
   - **Place.** Each piece goes where its photo was taken. Its depth is fitted to the solved points seen in that photo, which agree with SHARP's depth to about 1.5%. Splats strung between a thing and what is behind it, at depth edges, are dropped.
   - **Agree.** All pieces' depth is fused into one surface, and each piece is bent smoothly onto it, twice. Pieces move by about 2%, then 0.3%.
   - **Colour.** One brightness and colour correction per photo, from the points photos share, then a smooth correction toward the colour the photos agree on for each small cell of space.
   - **Doubt.** A photo's content in a cell is dropped when most other photos that see a surface there show a clearly different colour, or when two or more look straight through it. This is what removes a foot that stood on the rug for one photo.
   - **Own.** Each 4 cm cell of space goes to one photo: the nearest, most face-on one that covers it, with a cost for an owner the neighbouring cells do not share, so owners form patches.
   - **Thin.** Flat, evenly coloured blocks of a photo's grid become single discs lying on the surface. That takes 20 million splats to about 10 million without touching texture.
   - **Patch.** Where the surface is left with no splat, discs from the surface itself fill in.
5. **Mesh and collider** (`thread/mesh_photos.py --surface`). The agreed surface, cleaned and decimated.

The world is in metres as SHARP guesses them, so the floor and the up direction found from the surface are used directly.

**One photo** skips matching, solving and merging: its piece is the world, with the camera at the origin. It holds up from about half a metre around where the photo was taken. Beyond the photo's edge there is nothing.

## Measured

The bathroom (67 phone photos, 62 placed) on an M4 Pro with 64 GB:

| Step | Time | Peak memory |
|---|---|---|
| Match and solve | about 6.5 min | not measured |
| Pieces | about 6.5 min | 16 GB |
| Merge | about 3 min | 11 GB |
| Mesh and collider | about 1 min | not measured |
| Whole build | 15.7 min | 16 GB |

The result is about 9.8 million splats (550 MB) and renders at 60 frames a second in the viewer. One photo builds in about 25 seconds.

A second room, an office in 20 photos, built in 4.6 minutes with all 20 placed; its window glass comes out as holes.

Quality was judged by eye, on the same off-centre and close-up views of the old and the new build. No held-out score has been run on the new method yet.

## Known limits

- **White, glossy objects look chipped.** Where two photos' patches meet on a toilet lid or a bin, the photos disagree slightly about the object's shape and shade, and the join shows as shards. Larger patches, finer alignment and finer colour matching were each tried and did not remove it.
- **Thin dark things smear.** Cabinet handles, a tap and a curtain rod leave dark marks on what is behind them.
- **Fine texture is softer than the photos.** SHARP writes one splat pair per 2.5 photo pixels, so tile grout seen from across the room is soft.
- **A place no photo saw is a hole.** The viewer shows a muted tone of the room behind it, not black.
- **Mirrors and glass are not handled.**
- **The scale is SHARP's guess**, not a measurement.
- **SHARP's weights are licensed for non-commercial research only.** `PHOTOS_METHOD=train` builds without them.

## Tried and dropped

- Training longer at full size: worse, as above.
- Keeping one splat per 8 mm cell: lost all fine texture.
- Filling gaps with a photo's less certain splats: brought the haze back.
- Merging 2 by 2 splats into one and keeping the first one's orientation: blotchy. Discs laid on the surface work.
- A third, finer round of bending pieces onto the surface: more shards, not fewer.
- Moving splats exactly onto the agreed surface, and squashing them along its normal (`--flat` is still there): no cleaner, and thin dark things left specks.
- Larger owner patches, colour matched at a finer scale, a stronger preference for the photo that faces a surface: no visible change.

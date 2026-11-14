Blinded review of hair removal on 50 random validation dermoscopy photos (research only).
Each review_NNN.png shows three candidates X, Y, Z in random order. For every candidate fill
ratings_template.csv:
  hair_remaining_0to3   0 = no hair visible, 1 = faint, 2 = clear, 3 = heavy
  lesion_preserved_0to3 3 = outline, colour and texture look intact, 0 = lesion destroyed or altered
One candidate in each photo is the unprocessed image at the same resolution; it shows what hair
there was. Rate from the images alone and do not open the key file until finished. Note: DullRazor leaves blocky inpainting patches that may be recognisable, so blinding is imperfect; rate what you see.
Then run: python scripts/54_review_kit.py analyze --dir <this folder> --ratings <your csv>

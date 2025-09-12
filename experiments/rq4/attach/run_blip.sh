export IN_DIR=/path/to/your/images
export OUT_FILE=/path/to/output.tsv

python blip_inference.py --in_dir "$IN_DIR" --out_file "$OUT_FILE" --device cuda

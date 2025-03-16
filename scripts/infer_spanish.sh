#!/bin/bash

# Generate Spanish song
# Usage: ./scripts/infer_spanish.sh path/to/lrc path/to/reference_audio [output_directory]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <lrc_path> <ref_audio_path> [output_dir]"
    echo "Example: $0 infer/example/eg_es.lrc infer/example/eg_en.mp3 my_outputs"
    exit 1
fi

LRC_PATH=$1
REF_AUDIO_PATH=$2
OUTPUT_DIR=${3:-"infer/example/output"}

echo "Generating Spanish song from $LRC_PATH with reference audio $REF_AUDIO_PATH"
python infer/infer.py \
    --lrc-path "$LRC_PATH" \
    --ref-audio-path "$REF_AUDIO_PATH" \
    --chunked \
    --output-dir "$OUTPUT_DIR"

echo "Song generated and saved to $OUTPUT_DIR"
echo "Note: Since this is using our Spanish extension, the pronunciation quality should be evaluated by native Spanish speakers."
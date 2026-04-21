#!/usr/bin/env bash
# Convert all .MOV files under videos/ to H.264 MP4 for playback.
# Output: <original_name>.mp4 next to each source file.
# Skips files where the .mp4 already exists.

VIDEOS_DIR="$(dirname "$0")/videos"

# Pick encoder
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    VCODEC=h264_nvenc
    EXTRA1="-cq"; EXTRA2="23"; EXTRA3="-preset"; EXTRA4="p4"
else
    VCODEC=libx264
    EXTRA1="-crf"; EXTRA2="23"; EXTRA3="-preset"; EXTRA4="fast"
fi

echo "Encoder: $VCODEC"
echo ""

converted=0
skipped=0
failed=0

while IFS= read -r src; do
    dest="${src%.*}.mp4"

    if [ -f "$dest" ]; then
        echo "SKIP: $dest"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Converting: $src"
    if ffmpeg -hide_banner -loglevel error -y \
        -i "$src" \
        -vcodec "$VCODEC" "$EXTRA1" "$EXTRA2" "$EXTRA3" "$EXTRA4" \
        -acodec aac \
        "$dest"; then
        echo "  -> $dest"
        converted=$((converted + 1))
    else
        echo "  FAILED: $src"
        failed=$((failed + 1))
    fi

done < <(find "$VIDEOS_DIR" -type f -iname "*.mov")

echo ""
echo "Done. Converted: $converted  Skipped: $skipped  Failed: $failed"

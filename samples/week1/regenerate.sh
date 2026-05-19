#!/usr/bin/env bash
#
# Regenerate samples/week1/input.mp4 from scratch.
#
# This is the canonical test fixture for the week-1 spike. It's a 60-second
# English clip with a single speaker, generated via macOS `say` + ffmpeg so
# every checkout reproduces byte-for-byte. The actual Friday demo can use
# a real human voice — this fixture is for developers and CI.
#
# Why synthetic? Voice-cloning quality on a synthetic source is *artificially
# easy* (no noise, perfect pacing, no breathing). That's a known caveat; we
# keep a real-voice clip handy in samples/week1/real-voice/ for the demo.
#
# Usage:
#   ./samples/week1/regenerate.sh
#
# Requires: macOS (for `say`), ffmpeg.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v say >/dev/null 2>&1; then
  echo "error: this regeneration script needs macOS \`say\`. Skip on Linux." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found on PATH." >&2
  exit 1
fi

SCRIPT="The internet remembers everything. Every joke, every photo, every \
casual remark — preserved in some server somewhere. We used to think of the \
web as ephemeral, like a conversation that fades into the noise of the day. \
But the truth is closer to the opposite: nothing on the web ever really \
disappears. Old tweets, deleted comments, half-finished blog posts — \
they all live on, indexed, archived, occasionally surfaced years later as \
if no time had passed. That changes how we should think about what we say \
in public, and how we should treat each other when something old comes \
back into view. People grow. Posts don't. The kindest thing we can do for \
each other on a permanent web is to read old words with the charity we'd \
want extended to our own. To remember that a comment from years ago was \
written by a different version of a person, with different knowledge and \
different stakes. The web doesn't forget, but we can choose, every time, \
to be a little less merciless than the archive. That is the only way any \
of this stays livable."

# Produce a 16 kHz mono AIFF with `say`, then convert to MP4 with a still
# image for video. The still image is generated on the fly so this script
# has no external assets.
say -v "Samantha" -r 160 -o /tmp/linguacast-narration.aiff "$SCRIPT"

# Trim/pad to exactly 60s and convert to MP4. A solid-color frame at 720p
# keeps the file small while staying a real, playable MP4.
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x222233:s=1280x720:d=60:r=24" \
  -i /tmp/linguacast-narration.aiff \
  -t 60 \
  -c:v libx264 -preset veryfast -tune stillimage -pix_fmt yuv420p \
  -c:a aac -b:a 128k -ac 1 -ar 16000 \
  -shortest -movflags +faststart \
  input.mp4

rm -f /tmp/linguacast-narration.aiff
echo "wrote $(pwd)/input.mp4"
ffprobe -hide_banner -loglevel error -show_entries \
  stream=codec_type,duration,sample_rate,channels -of default=nw=1 input.mp4

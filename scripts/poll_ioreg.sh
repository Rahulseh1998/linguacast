#!/usr/bin/env bash
# Poll ioreg "Alloc system memory" + total system memory pressure every 2s.
# Usage: poll_ioreg.sh <output.tsv>
# Columns: epoch_s  alloc_bytes  app_memory_bytes  compressed_bytes  wired_bytes  free_bytes
set -euo pipefail
out="${1:-ioreg.tsv}"
echo -e "epoch_s\talloc_bytes\tapp_mem_bytes\tcompressed_bytes\twired_bytes\tfree_bytes" > "$out"
while true; do
  ts=$(date +%s)
  alloc=$(ioreg -r -c IOAccelerator 2>/dev/null | awk -F'"Alloc system memory"=' 'NF>1 {split($2,a,","); print a[1]; exit}')
  alloc="${alloc:-0}"
  # vm_stat output is in 16384-byte (page-size on Apple Silicon) units; convert.
  read app compressed wired freep <<<"$(vm_stat | awk '
    /App Memory/             { gsub(/\./,"",$3); app=$3 }
    /Pages occupied by compressor/ { gsub(/\./,"",$5); comp=$5 }
    /Pages wired down/       { gsub(/\./,"",$4); wired=$4 }
    /Pages free/             { gsub(/\./,"",$3); freep=$3 }
    END { print app, comp, wired, freep }')"
  ps=16384
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$alloc" "$((app*ps))" "$((compressed*ps))" "$((wired*ps))" "$((freep*ps))" >> "$out"
  sleep 2
done

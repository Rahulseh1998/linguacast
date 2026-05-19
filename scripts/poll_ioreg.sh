#!/usr/bin/env bash
out="${1:-ioreg.tsv}"
printf "epoch_s\talloc_bytes\tactive_bytes\tcompressed_bytes\twired_bytes\tfree_bytes\n" > "$out"
PS=16384
while :; do
  ts=$(date +%s)
  alloc=$(ioreg -r -c IOAccelerator 2>/dev/null | awk -F'"Alloc system memory"=' 'NF>1 {split($2,a,","); print a[1]; exit}')
  vm=$(vm_stat 2>/dev/null | awk -v ps=$PS '
    /^Pages active/                { gsub(/\./,"",$3); active=$3*ps }
    /Pages occupied by compressor/ { gsub(/\./,"",$5); comp=$5*ps }
    /^Pages wired down/            { gsub(/\./,"",$4); wired=$4*ps }
    /^Pages free/                  { gsub(/\./,"",$3); freep=$3*ps }
    END { printf "%s\t%s\t%s\t%s\n", active+0, comp+0, wired+0, freep+0 }')
  printf "%s\t%s\t%s\n" "$ts" "${alloc:-0}" "$vm" >> "$out"
  sleep 2
done

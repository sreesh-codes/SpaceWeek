#!/bin/zsh
# Run 05_jamming.py one day at a time and delete each day's raw JSON (~100 MB/day)
# after parsing, so the full window fits on a nearly full disk.
set -e
cd "$(dirname "$0")"
mkdir -p data/jamming/days
d=${1:-2026-09-01}; end=${2:-2026-10-31}
while [[ ! "$d" > "$end" ]]; do
  if [[ ! -s data/jamming/days/$d.csv ]]; then
    ./venv-jam/bin/python 05_jamming.py --start $d --end $d | tail -1
    mv data/jamming/gulf_jamming_cells.csv data/jamming/days/$d.csv
    rm -rf downloaded_json_files/jamming
  fi
  d=$(date -j -v+1d -f %Y-%m-%d $d +%Y-%m-%d)
done
f=(data/jamming/days/*.csv)
{ head -1 $f[1]; for x in $f; do tail -n +2 $x; done } > data/jamming/gulf_jamming_cells.csv
echo "merged $(($(wc -l < data/jamming/gulf_jamming_cells.csv)-1)) rows from ${#f} days"

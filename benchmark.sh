#!/bin/bash
set -e

SIZES="64 128 256 512 1024"
SMOOTHERS="jacobi jacobi_amortecido gauss_seidel gauss_seidel_rb sor"
RUNS=5
BIN="./multigrid_2d"

echo "Compilando..." >&2
g++ -O3 -o "$BIN" cpu/multigrid.cpp -lm
echo "OK" >&2

echo "smoother,n,run,vcycles,tempo_ms,residuo,erro_max"

for smoother in $SMOOTHERS; do
  for n in $SIZES; do
    for run in $(seq 1 $RUNS); do
      echo "  $smoother n=$n run=$run" >&2
      output=$($BIN --n "$n" --smoother "$smoother" 2>&1)

      vcycles=$(echo "$output" | awk '/Convergiu em/{print $3}')
      [ -z "$vcycles" ] && vcycles=200

      tempo=$(echo "$output" | awk '/tempo total:/{print $3}')
      residuo=$(echo "$output" | awk '/residuo final:/{print $3}')
      erro=$(echo "$output" | awk '/erro maximo:/{print $3}')

      echo "$smoother,$n,$run,$vcycles,$tempo,$residuo,$erro"
    done
  done
done

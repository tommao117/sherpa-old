#!/usr/bin/env bash
# Build libsherpa_cupti.so. Usage: build.sh <cuda_root> <out_dir>
set -euo pipefail
CUDA_ROOT="$1"; OUT="$2"
mkdir -p "$OUT"
gcc -O2 -shared -fPIC -fvisibility=hidden \
  -I"$CUDA_ROOT/include" "$(dirname "$0")/cupti_trace.c" \
  -L"$CUDA_ROOT/lib" -Wl,-rpath,"$CUDA_ROOT/lib" -lcupti -lpthread \
  -o "$OUT/libsherpa_cupti.so"
echo "$OUT/libsherpa_cupti.so"

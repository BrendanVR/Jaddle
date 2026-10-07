#!/usr/bin/env bash
# Build the glop_presolve helper binary used by jaddle/glop_helpers.py.
#
# Downloads the OR-Tools C++ binary release matching the installed `ortools`
# Python package into third_party/ (gitignored) and builds against it. The
# Python bindings don't expose glop's presolver, hence the separate binary.
#
# Usage:  tools/glop_presolve/build.sh
#         ORTOOLS_DIR=/path/to/or-tools tools/glop_presolve/build.sh  # use an existing install
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

if [[ -z "${ORTOOLS_DIR:-}" ]]; then
  OT_VERSION="${ORTOOLS_VERSION:-$(python -c 'import ortools; print(ortools.__version__)')}"
  MAJOR_MINOR="${OT_VERSION%.*}"
  OS_ID="$(. /etc/os-release && echo "$ID")"
  OS_VERSION_ID="$(. /etc/os-release && echo "$VERSION_ID")"
  # The tarball's top-level directory (or-tools_x86_64_Ubuntu-24.04_...) is
  # named differently from the asset, so glob for it by version.
  find_install() {
    compgen -G "$REPO_ROOT/third_party/or-tools_*_cpp_v${OT_VERSION}" | head -n1 || true
  }
  ORTOOLS_DIR="$(find_install)"
  if [[ -z "$ORTOOLS_DIR" ]]; then
    mkdir -p "$REPO_ROOT/third_party"
    URL="https://github.com/google/or-tools/releases/download/v${MAJOR_MINOR}/or-tools_amd64_${OS_ID}-${OS_VERSION_ID}_cpp_v${OT_VERSION}.tar.gz"
    echo "Downloading $URL"
    curl -fL "$URL" | tar -xz -C "$REPO_ROOT/third_party"
    ORTOOLS_DIR="$(find_install)"
  fi
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ORTOOLS_DIR"
cmake --build "$HERE/build" -j
echo "Built $HERE/build/glop_presolve"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${ROOT_DIR}/requirements.txt"
WHEELHOUSE="${ROOT_DIR}/vendor/wheels"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python binary '${PYTHON_BIN}' not found. Set PYTHON_BIN=<python> and retry." >&2
  exit 1
fi

run_pip() {
  "${PYTHON_BIN}" -m pip "$@"
}

have_all_wheels() {
  local missing=0
  local names
  names=$(awk -F'[<>= ]+' 'NF>0 && $1 !~ /^#/ {print tolower($1)}' "${REQ_FILE}")
  while IFS= read -r pkg; do
    [[ -z "${pkg}" ]] && continue
    if ! find "${WHEELHOUSE}" -maxdepth 1 -type f \( -iname "${pkg}-*.whl" -o -iname "${pkg//-/_}-*.whl" \) | head -n 1 | grep -q .; then
      echo "Missing wheel for ${pkg} in ${WHEELHOUSE}" >&2
      missing=1
    fi
  done <<<"${names}"
  return "${missing}"
}

echo "==> Upgrading pip tooling (best effort)"
run_pip install --upgrade pip wheel setuptools >/dev/null 2>&1 || true

if [ -d "${WHEELHOUSE}" ] && [ "$(find "${WHEELHOUSE}" -maxdepth 1 -name '*.whl' | wc -l)" -gt 0 ]; then
  echo "==> Found local wheelhouse at ${WHEELHOUSE}; attempting offline install"
  if have_all_wheels; then
    run_pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"
    echo "✅ Dependencies installed from local wheelhouse."
    exit 0
  else
    echo "⚠️ Wheelhouse exists but is incomplete; will try online install next."
  fi
fi

echo "==> Attempting online install via configured pip index/proxy"
if run_pip install -r "${REQ_FILE}"; then
  echo "✅ Dependencies installed from package index."
  exit 0
fi

cat >&2 <<'MSG'

❌ Dependency installation failed (likely blocked network/proxy or no reachable package index).

Use one of these reliable fixes:

1) Configure a reachable internal mirror before install:
   export PIP_INDEX_URL="https://<your-internal-pypi>/simple"
   export PIP_TRUSTED_HOST="<your-internal-pypi-host>"
   ./scripts/install_deps.sh

2) Build an offline wheel bundle on a machine with internet, then copy to this repo:
   pip download -r requirements.txt -d vendor/wheels
   # copy vendor/wheels into this repository
   ./scripts/install_deps.sh

3) If corporate proxy is required, export proxy variables and retry:
   export HTTPS_PROXY="http://<proxy-host>:<port>"
   export HTTP_PROXY="http://<proxy-host>:<port>"
   ./scripts/install_deps.sh
MSG

exit 1

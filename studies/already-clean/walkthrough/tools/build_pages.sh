#!/usr/bin/env bash
# Copy the walkthrough to a clean gh-pages tree, then check the copy (BUILD.md §4).
# The copy holds only what the browser needs: .html, .css, .js (data.js included) and images; no Python, no
# Markdown, no plans, no tools. Then it runs both tools' self-tests, the vocabulary scan and the number check on the
# copy, and stops at the first failure.
#
#   tools/build_pages.sh <output directory>     the directory must not exist yet, or be empty
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$(dirname "$here")"
repo="$(git -C "$src" rev-parse --show-toplevel)"
out="${1:?usage: tools/build_pages.sh <output directory>}"

mkdir -p "$out"
out="$(cd "$out" && pwd)"
case "$out/" in "$src"/*) echo "build_pages: the output directory must be outside $src" >&2; exit 1;; esac
if [ -n "$(ls -A "$out")" ]; then echo "build_pages: $out is not empty" >&2; exit 1; fi

cd "$src"
find . -path ./tools -prune -o -type f \( -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.png' \
  -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.gif' -o -name '*.svg' -o -name '*.webp' -o -name '*.ico' \) -print0 |
while IFS= read -r -d '' f; do
  mkdir -p "$out/$(dirname "$f")"
  cp "$f" "$out/$f"
done

# A data.js still stored as a Git LFS pointer would publish a page with no data.
if grep -rl --include=data.js '^version https://git-lfs' "$out" >/dev/null; then
  echo "build_pages: these data.js files are Git LFS pointers; run git lfs pull:" >&2
  grep -rl --include=data.js '^version https://git-lfs' "$out" >&2
  exit 1
fi

python3 "$here/scan_vocab.py" --self-test
python3 "$here/scan_vocab.py" "$out"
python3 "$here/check_numbers.py" --self-test
python3 "$here/check_numbers.py" "$out" --repo "$repo"
echo "build_pages: $out holds $(find "$out" -type f | wc -l | tr -d ' ') files and passed both checks"

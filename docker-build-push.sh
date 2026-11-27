#!/usr/bin/env bash
# Builds the muvis-align-xpra image and pushes it to quay.io, tagged with the
# current GitHub release version (and "latest").
#
# Usage: ./docker-build-push.sh [version]
#   version   optional, e.g. v0.4.3 - defaults to the latest GitHub release tag
#
# Requires: docker (logged in via `docker login quay.io`), and either `gh`
# (authenticated) or plain network access to api.github.com as a fallback.

set -euo pipefail

REPO="ccp-volume-em/muvis-align"
IMAGE="quay.io/ccp-volume-em/muvis-align-xpra"
TARGET="muvis-align-xpra"

get_latest_release() {
    if command -v gh >/dev/null 2>&1; then
        gh release view --repo "$REPO" --json tagName -q .tagName
    else
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/'
    fi
}

VERSION="${1:-$(get_latest_release)}"
if [ -z "$VERSION" ]; then
    echo "ERROR: could not determine version - pass one explicitly, e.g. ./docker-build-push.sh v0.4.3" >&2
    exit 1
fi

echo "Building ${IMAGE}:${VERSION} (target: ${TARGET}) ..."
# the package states the release's version (v0.7.0 -> 0.7.0): the image holds only part of the work tree
docker build --target "$TARGET" --build-arg "MUVIS_ALIGN_VERSION=${VERSION#v}" \
    -t "${IMAGE}:${VERSION}" -t "${IMAGE}:latest" .

echo "Pushing ${IMAGE}:${VERSION} and ${IMAGE}:latest ..."
# only the tags built here: --all-tags would also push any stale local tag of the image
docker push "${IMAGE}:${VERSION}"
docker push "${IMAGE}:latest"

echo "Done: ${IMAGE}:${VERSION} (also tagged latest)"

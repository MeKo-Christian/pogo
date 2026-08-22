#!/bin/bash
set -euo pipefail

# treefmt cannot be installed with `go install`: the v2 module zip contains a
# path with an emoji (test/examples/emoji 🕰️/README.md), which the Go module
# format rejects. Fetch the release binary instead.

TREEFMT_VERSION="${TREEFMT_VERSION:-2.5.0}"
BIN_DIR="${BIN_DIR:-$(go env GOPATH)/bin}"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
x86_64)
	ARCH="amd64"
	;;
aarch64 | arm64)
	ARCH="arm64"
	;;
*)
	echo "Unsupported architecture: $ARCH" >&2
	exit 1
	;;
esac

case "$OS" in
linux | darwin) ;;
*)
	echo "Unsupported OS: $OS" >&2
	exit 1
	;;
esac

TARBALL="treefmt_${TREEFMT_VERSION}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/numtide/treefmt/releases/download/v${TREEFMT_VERSION}/${TARBALL}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading treefmt from $URL..."
curl -sSfL -o "$TMP_DIR/$TARBALL" "$URL"
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR" treefmt

mkdir -p "$BIN_DIR"
install -m 0755 "$TMP_DIR/treefmt" "$BIN_DIR/treefmt"
echo "Installed treefmt $TREEFMT_VERSION to $BIN_DIR/treefmt"

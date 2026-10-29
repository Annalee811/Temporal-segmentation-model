#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec docker run --rm --network none --mount "type=bind,source=$root,target=$root" --workdir "$root" --entrypoint python root-tf112:local "$@"

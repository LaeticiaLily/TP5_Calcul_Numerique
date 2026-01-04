#!/usr/bin/env bash
docker run -it --rm \
  --hostname fedora \
  -v "$PWD:/app" \
  poisson-tp

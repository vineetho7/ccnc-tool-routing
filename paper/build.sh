#!/bin/bash
# Build the submission PDF.
cd "$(dirname "$0")"
/opt/homebrew/bin/tectonic -X compile confusable_telecom_apis.tex "$@"

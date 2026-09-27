#!/usr/bin/env python3
"""Compatibility import and CLI alias for the rolling four-GW certifier.

The canonical production entrypoint is scripts/certify_four_gw.py. This historic
module remains so existing callers can import its tested PE-9 lifecycle helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from certify_four_gw import *  # noqa: F401,F403 - preserve the historical helper API
from certify_four_gw import main


if __name__ == "__main__":
    raise SystemExit(main())

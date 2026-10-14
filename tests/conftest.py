"""Shared test configuration."""

import os

# The API's rate limiter is shared by every test in the run; keep it out of the way.
# Rate limiting itself is covered in test_security.py with a dedicated app.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")

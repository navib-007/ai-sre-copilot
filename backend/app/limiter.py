"""
limiter.py — Global Rate Limiter Instance
========================================
CONCEPT: Preventing circular imports while sharing the SlowAPI Limiter across routes.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize the rate limiter using the client's IP address
limiter = Limiter(key_func=get_remote_address)

"""
tools/logs_tool.py — Application Log Search Tool (Simulated)
=============================================================
CONCEPT: Simulated Tools for Learning

  In real production, this tool would connect to:
    - Elasticsearch / OpenSearch
    - Datadog / Splunk / Loki
    - AWS CloudWatch Logs

  For this learning project, we SIMULATE realistic log data.

  WHY SIMULATE instead of connecting to real services?
    1. No external dependencies (everyone can run this)
    2. Controlled data = predictable agent behavior for learning
    3. Focus on the AGENT LOGIC, not the integration plumbing
    4. Realistic enough to understand real-world patterns

CONCEPT: Realistic Log Simulation
  We generate logs that look like real production logs:
    - Structured format (timestamp, level, service, message)
    - Realistic error patterns (OOM kills, connection timeouts, pod crashes)
    - Time-correlated: errors cluster around "incidents" in the timeline
    - Service-specific: different services have different failure modes

  The agent's job is to INTERPRET these simulated logs, just like
  it would with real logs. The simulation teaches the pattern.

CONCEPT: Log Analysis in Incident Investigation
  When investigating an incident, the Incident Agent asks:
    1. "What errors appeared in the logs around the incident time?"
    2. "Are there patterns? (repeated errors, cascading failures?)"
    3. "Which service threw the first error?" (blast radius analysis)

  The agent uses log data as EVIDENCE to feed into the RCA Agent.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Input Schema ─────────────────────────────────────────────────────────────

class LogSearchInput(BaseModel):
    """Input schema for log search."""
    service: str = Field(
        description=(
            "Name of the service to search logs for. "
            "Examples: 'payment-service', 'auth-service', 'api-gateway', "
            "'postgres', 'redis', 'kubernetes', 'nginx'."
        )
    )
    time_window: str = Field(
        default="1h",
        description=(
            "Time window to search. Format: '<number><unit>' where unit is "
            "'m' (minutes), 'h' (hours), or 'd' (days). "
            "Examples: '30m', '1h', '6h', '24h', '7d'."
        )
    )
    log_level: Optional[str] = Field(
        default=None,
        description="Filter by log level: 'ERROR', 'WARN', 'INFO'. Leave null for all levels.",
    )
    keyword: Optional[str] = Field(
        default=None,
        description="Optional keyword to search within log messages. E.g., 'OOMKilled', 'timeout', 'connection refused'.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of log entries to return.",
    )


# ─── Simulated Log Patterns ───────────────────────────────────────────────────
# CONCEPT: Realistic failure patterns for each service type
# These templates simulate real-world production log patterns

_LOG_PATTERNS = {
    "payment-service": {
        "ERROR": [
            "OOMKilled: container exceeded memory limit (2Gi). Pod terminated.",
            "Connection refused to postgres://payment-db:5432 after 3 retries",
            "Transaction timeout after 30000ms: checkout request ID {req_id}",
            "Unhandled exception in PaymentProcessor: NullPointerException at line 247",
            "Circuit breaker OPEN for downstream service: fraud-detection-api",
            "Database connection pool exhausted (max=50, active=50, idle=0)",
        ],
        "WARN": [
            "High memory usage: 1.8Gi / 2Gi (90%). Consider scaling.",
            "Slow query detected: SELECT * FROM transactions took 4523ms",
            "Retry attempt 2/3 for payment gateway API call",
            "Connection pool usage high: 45/50 active connections",
        ],
        "INFO": [
            "Payment processed successfully: order_id={req_id} amount=$149.99",
            "Health check passed. DB: connected, Cache: connected",
            "Pod started. Memory limit: 2Gi, CPU limit: 500m",
        ],
    },
    "auth-service": {
        "ERROR": [
            "Redis connection timeout after 5000ms: ECONNRESET",
            "JWT validation failed: token expired for user_id={req_id}",
            "Database write failed: too many connections (max=100)",
            "Failed to refresh access token: upstream OAuth provider returned 503",
        ],
        "WARN": [
            "High login failure rate: 47 failures in last 60 seconds",
            "Redis latency spike: avg 450ms (threshold: 100ms)",
            "Session store approaching capacity: 95% full",
        ],
        "INFO": [
            "User login successful: user_id={req_id}",
            "Token refreshed for user_id={req_id}",
            "Service started on port 8080",
        ],
    },
    "api-gateway": {
        "ERROR": [
            "Upstream timeout: payment-service did not respond within 30s",
            "Circuit breaker triggered for route /api/payments (error rate: 67%)",
            "SSL certificate validation failed for downstream service",
            "Rate limit exceeded: IP 192.168.1.{req_id} blocked for 60s",
        ],
        "WARN": [
            "Increased error rate on /api/checkout: 12% (threshold: 5%)",
            "High request queue depth: 847 requests waiting",
            "Downstream service payment-service response time: 8.2s (SLA: 2s)",
        ],
        "INFO": [
            "Request routed to payment-service: GET /api/payments",
            "Health check: all upstream services operational",
            "Configuration reloaded: 5 routes updated",
        ],
    },
    "postgres": {
        "ERROR": [
            "FATAL: too many connections (max_connections=100, current=103)",
            "ERROR: deadlock detected on relation 'transactions'",
            "PANIC: could not write to file 'pg_wal': No space left on device",
            "ERROR: invalid page in block 8472 of relation base/16384/payment_transactions",
        ],
        "WARN": [
            "LOG: checkpoint taking 42.3s, write 38.1s sync 4.2s",
            "WARNING: connection pool saturation at 94%",
            "SLOW QUERY: duration 12345ms — SELECT * FROM transactions WHERE ...",
        ],
        "INFO": [
            "LOG: database system is ready to accept connections",
            "LOG: checkpoint complete: wrote 3847 buffers (23.5%)",
            "LOG: autovacuum: processing relation 'public.tickets'",
        ],
    },
    "kubernetes": {
        "ERROR": [
            "0/3 nodes available: insufficient memory. Preempting pods to free resources.",
            "Pod payment-service-7d4f8b9c6-xkj2p: OOMKilled (exit code 137)",
            "Liveness probe failed: connection refused on port 8080",
            "Pod failed to schedule: node selector does not match any node",
        ],
        "WARN": [
            "Node worker-node-2 is NotReady: kubelet stopped posting node status",
            "HorizontalPodAutoscaler scaled payment-service to 5 replicas (CPU: 87%)",
            "Pod payment-service-7d4f8b9c6-xkj2p restarted 3 times in last 10 minutes",
        ],
        "INFO": [
            "Deployment payment-service updated: 3/3 replicas ready",
            "Service payment-service ClusterIP 10.96.0.142 created",
            "ConfigMap payment-config updated",
        ],
    },
}

# Default patterns for unknown services
_DEFAULT_PATTERNS = {
    "ERROR": [
        "Unexpected error occurred in service",
        "Connection to dependency failed",
        "Service health check failed",
    ],
    "WARN": [
        "High resource utilization detected",
        "Slow response time from upstream",
        "Retry attempt failed",
    ],
    "INFO": [
        "Service operating normally",
        "Request processed successfully",
        "Configuration loaded",
    ],
}


def _parse_time_window(time_window: str) -> int:
    """Parse time window string into minutes. Returns 60 for invalid inputs."""
    try:
        unit = time_window[-1].lower()
        value = int(time_window[:-1])
        if unit == "m":
            return value
        elif unit == "h":
            return value * 60
        elif unit == "d":
            return value * 60 * 24
    except (ValueError, IndexError):
        pass
    return 60  # Default to 1 hour


def _generate_log_entries(
    service: str,
    minutes: int,
    log_level: Optional[str],
    keyword: Optional[str],
    limit: int,
) -> list[dict]:
    """
    Generate realistic simulated log entries for a service.

    CONCEPT: Weighted Random Generation
      Real logs have many more INFO than WARN, many more WARN than ERROR.
      We use weighted probability to reflect this:
        ERROR: 15% of logs (when no filter)
        WARN:  25% of logs
        INFO:  60% of logs

      When filtering by ERROR, we only return ERROR-level entries.
    """
    patterns = _LOG_PATTERNS.get(service.lower(), _DEFAULT_PATTERNS)
    now = datetime.now(timezone.utc)
    entries = []

    # Determine which levels to generate
    if log_level:
        levels_weights = [(log_level.upper(), 1.0)]
    else:
        levels_weights = [("ERROR", 0.15), ("WARN", 0.25), ("INFO", 0.60)]

    # Generate log entries distributed across the time window
    num_to_generate = min(limit * 3, 150)  # Generate extra, then filter
    for i in range(num_to_generate):
        # Pick level based on weights
        level = random.choices(
            [lw[0] for lw in levels_weights],
            weights=[lw[1] for lw in levels_weights],
            k=1,
        )[0]

        available_messages = patterns.get(level, _DEFAULT_PATTERNS.get(level, ["Unknown event"]))
        message = random.choice(available_messages).format(req_id=random.randint(1000, 9999))

        # Apply keyword filter
        if keyword and keyword.lower() not in message.lower():
            continue

        # Random timestamp within window (more recent timestamps more likely)
        offset_minutes = random.randint(0, minutes)
        timestamp = now - timedelta(minutes=offset_minutes)

        entries.append({
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": level,
            "service": service,
            "message": message,
        })

        if len(entries) >= limit:
            break

    # Sort by timestamp descending (most recent first)
    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    return entries[:limit]


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_logs_tool() -> StructuredTool:
    """
    Returns the log search tool (no DB needed — data is simulated).

    CONCEPT: Stateless Tools
      Unlike ticket/incident tools that need a DB session, this tool
      generates data in-memory. It's stateless — no external dependencies.
      This makes it fast, testable, and dependency-free.

    Returns:
        A configured StructuredTool for log searching.
    """

    def search_logs(
        service: str,
        time_window: str = "1h",
        log_level: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """
        Search application logs for a specific service within a time window.

        Use this tool during incident investigation to:
        - Find error messages and stack traces
        - Identify when errors started (time correlation)
        - Find patterns in failures (repeated errors = likely root cause)
        - Look for resource exhaustion messages (OOM, disk full, max connections)

        ⚠️ Note: This tool uses simulated log data for learning purposes.
        In production, this would connect to your log aggregation system
        (Elasticsearch, Datadog, Splunk, CloudWatch Logs, etc.)

        Returns formatted log entries with timestamps, levels, and messages.
        """
        logger.info(
            "logs_tool_invoked",
            service=service,
            time_window=time_window,
            log_level=log_level,
            keyword=keyword,
        )

        minutes = _parse_time_window(time_window)
        entries = _generate_log_entries(
            service=service,
            minutes=minutes,
            log_level=log_level,
            keyword=keyword,
            limit=limit,
        )

        if not entries:
            return (
                f"No {log_level or ''} log entries found for service '{service}' "
                f"in the last {time_window}"
                + (f" matching keyword '{keyword}'" if keyword else "")
                + ".\n[Note: Using simulated log data]"
            )

        # Format entries for LLM consumption
        lines = [
            f"Log entries for '{service}' (last {time_window})"
            + (f" | Level: {log_level}" if log_level else "")
            + (f" | Keyword: '{keyword}'" if keyword else "")
            + f"\n[Note: Simulated data — {len(entries)} entries shown]\n"
        ]
        for entry in entries:
            level_emoji = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🟢"}.get(entry["level"], "⚪")
            lines.append(
                f"{level_emoji} [{entry['timestamp']}] {entry['level']} — {entry['message']}"
            )

        # Add summary statistics
        error_count = sum(1 for e in entries if e["level"] == "ERROR")
        warn_count = sum(1 for e in entries if e["level"] == "WARN")
        lines.append(f"\nSummary: {error_count} ERROR(s), {warn_count} WARN(s) in this window.")

        return "\n".join(lines)

    return StructuredTool.from_function(
        func=search_logs,      # Sync function (no DB calls)
        name="search_logs",
        description=(
            "Search application logs for a service to find errors, warnings, and events. "
            "Use during incident investigation to identify error patterns, find when issues started, "
            "and look for resource exhaustion (OOM, disk full, max connections). "
            "Key services: 'payment-service', 'auth-service', 'api-gateway', 'postgres', 'kubernetes'."
        ),
        args_schema=LogSearchInput,
    )

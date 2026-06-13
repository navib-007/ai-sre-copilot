"""
db/seed.py — Sample Data Seeder
================================
CONCEPT: Database Seeding

  Seed data gives you realistic test data so you can:
  1. Immediately test the API without manually creating records
  2. Develop against realistic data
  3. Run automated tests with known state

  We check before inserting to make seeding idempotent
  (running it multiple times doesn't create duplicates).

Run with:
  python -m app.db.seed
"""

import asyncio
import sys
from pathlib import Path

# Ensure we can import from backend/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Incident, Ticket, User
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ─── Sample Users ─────────────────────────────────────────────────────────────
SAMPLE_USERS = [
    {
        "username": "admin",
        "email": "admin@opsplatform.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "admin",
        "is_active": True,
    },
    {
        "username": "alice_engineer",
        "email": "alice@opsplatform.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "engineer",
        "is_active": True,
    },
    {
        "username": "bob_viewer",
        "email": "bob@opsplatform.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "viewer",
        "is_active": True,
    },
]

# ─── Sample Tickets ───────────────────────────────────────────────────────────
SAMPLE_TICKETS = [
    {
        "title": "Database backup failing on prod-db-01",
        "description": (
            "The nightly database backup job for prod-db-01 has been failing since 2024-01-10. "
            "Error: 'No space left on device'. The /backup mount point is at 98% capacity. "
            "Last successful backup was 3 days ago."
        ),
        "status": "open",
        "priority": "high",
        "category": "database",
        "created_by": 2,
    },
    {
        "title": "Kubernetes pod stuck in CrashLoopBackOff - payment-api",
        "description": (
            "The payment-api pod in production has been in CrashLoopBackOff state for 45 minutes. "
            "kubectl logs show: 'Error: ECONNREFUSED - Cannot connect to Redis'. "
            "This is blocking all payment processing."
        ),
        "status": "in_progress",
        "priority": "critical",
        "category": "kubernetes",
        "created_by": 2,
        "assigned_to": 2,
    },
    {
        "title": "SSL certificate expiring in 7 days for api.example.com",
        "description": (
            "The SSL certificate for api.example.com will expire on 2024-01-20. "
            "Need to renew via Let's Encrypt before it causes service disruption. "
            "Auto-renewal appears to have failed last cycle."
        ),
        "status": "open",
        "priority": "high",
        "category": "security",
        "created_by": 1,
    },
    {
        "title": "Memory usage on auth-service exceeding 85% threshold",
        "description": (
            "Monitoring alert: auth-service memory usage has been above 85% for 2 hours. "
            "Current: 87%. Memory limit: 512Mi. Service is still functional but at risk. "
            "Suspected memory leak introduced in v2.3.1 deployment yesterday."
        ),
        "status": "open",
        "priority": "medium",
        "category": "kubernetes",
        "created_by": 2,
    },
    {
        "title": "Elasticsearch index size growing too fast",
        "description": (
            "The application-logs Elasticsearch index is growing at 50GB/day, "
            "significantly faster than the projected 10GB/day. "
            "At current rate, disk will fill in 3 days. Need to investigate cause and "
            "implement log retention policy."
        ),
        "status": "open",
        "priority": "medium",
        "category": "monitoring",
        "created_by": 1,
    },
    {
        "title": "Slow API response times on /api/reports endpoint",
        "description": (
            "Users reporting that /api/reports is taking 15-30 seconds to respond. "
            "This endpoint generates PDF reports from database aggregations. "
            "Normal response time is 2-3 seconds. Started after the DB index changes last week."
        ),
        "status": "resolved",
        "priority": "high",
        "category": "performance",
        "created_by": 2,
        "assigned_to": 2,
        "resolution": "Added missing composite index on (user_id, created_at). Response time back to 1.8s.",
    },
]

# ─── Sample Incidents ─────────────────────────────────────────────────────────
SAMPLE_INCIDENTS = [
    {
        "title": "P1: Payment Service Complete Outage",
        "description": (
            "The payment-service is completely unresponsive. All payment transactions are failing. "
            "Error rate is 100%. First detected at 14:32 UTC by synthetic monitoring. "
            "Affecting all customers in production environment."
        ),
        "severity": "P1",
        "status": "resolved",
        "affected_services": "payment-service,checkout-api,order-service",
        "root_cause": (
            "Memory leak in payment-service v2.5.0 caused OOM kill under high traffic. "
            "The leak was in the PDF receipt generation module (introduced in v2.5.0). "
            "Fix: Rolled back to v2.4.2 and increased memory limit from 512Mi to 1Gi."
        ),
        "reported_by": 2,
    },
    {
        "title": "P2: Auth Service Degraded Performance",
        "description": (
            "Authentication service showing 3x normal latency (avg 2.1s vs normal 0.7s). "
            "Login success rate dropped to 78%. Customers experiencing intermittent login failures."
        ),
        "severity": "P2",
        "status": "investigating",
        "affected_services": "auth-service,user-api",
        "reported_by": 2,
    },
    {
        "title": "P3: Database Backup Job Failure",
        "description": (
            "The automated nightly backup job for prod-db-01 failed. "
            "No backup created for 2024-01-14. All data is intact, just the backup failed."
        ),
        "severity": "P3",
        "status": "mitigated",
        "affected_services": "prod-db-01",
        "root_cause": "Insufficient disk space on backup volume. Extended volume by 500GB.",
        "reported_by": 1,
    },
]


async def seed_users(session: AsyncSession) -> dict[str, int]:
    """Create sample users. Returns {username: user_id} mapping."""
    user_ids: dict[str, int] = {}

    for user_data in SAMPLE_USERS:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.username == user_data["username"])
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info("user_already_exists", username=user_data["username"])
            user_ids[user_data["username"]] = existing.id
        else:
            user = User(**user_data)
            session.add(user)
            await session.flush()
            user_ids[user_data["username"]] = user.id
            logger.info("user_created", username=user.username, id=user.id, role=user.role)

    return user_ids


async def seed_tickets(session: AsyncSession) -> None:
    """Create sample tickets if none exist."""
    count_result = await session.execute(select(Ticket))
    if count_result.scalars().first() is not None:
        logger.info("tickets_already_seeded_skipping")
        return

    for ticket_data in SAMPLE_TICKETS:
        ticket = Ticket(**ticket_data)
        session.add(ticket)
        await session.flush()
        logger.info(
            "ticket_created",
            id=ticket.id,
            title=ticket.title[:40],
            priority=ticket.priority,
        )


async def seed_incidents(session: AsyncSession) -> None:
    """Create sample incidents if none exist."""
    count_result = await session.execute(select(Incident))
    if count_result.scalars().first() is not None:
        logger.info("incidents_already_seeded_skipping")
        return

    for incident_data in SAMPLE_INCIDENTS:
        incident = Incident(**incident_data)
        session.add(incident)
        await session.flush()
        logger.info(
            "incident_created",
            id=incident.id,
            severity=incident.severity,
            title=incident.title[:40],
        )


async def run_seed() -> None:
    """Run all seeders."""
    setup_logging()

    logger.info("seeding_started")

    # Initialize DB (create tables if they don't exist)
    await init_db()

    async with AsyncSessionLocal() as session:
        try:
            await seed_users(session)
            await seed_tickets(session)
            await seed_incidents(session)
            await session.commit()
            logger.info("seeding_completed_successfully")
        except Exception as e:
            await session.rollback()
            logger.error("seeding_failed", error=str(e))
            raise


if __name__ == "__main__":
    asyncio.run(run_seed())

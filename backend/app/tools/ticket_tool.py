"""
tools/ticket_tool.py — Ticket CRUD Agent Tools
================================================
CONCEPT: Data-Mutating Tools in Agents

  Unlike RAG (read-only), ticket tools WRITE to the database.
  This raises important design questions:
    1. Who authorizes the write? (Phase 9: JWT/RBAC)
    2. What if the write fails mid-operation?
    3. Should destructive actions require approval? (Phase 6: HITL)

  For Phase 4, we keep it simple: tools write directly.
  Phase 6 will add approval gates for high-risk operations.

CONCEPT: Structured Output from LLM (Extraction)
  The Ticket Agent receives natural language like:
    "Create a high-priority ticket for the Redis connection timeout
     issue in the auth-service. It started after the Redis upgrade."

  The LLM must EXTRACT structured fields from this text:
    title:       "Redis connection timeout in auth-service"
    priority:    "high"
    category:    "database"
    description: "Redis connection timeout after upgrade..."

  This is "information extraction" — turning unstructured text into
  structured data. The tool's Pydantic schema guides the LLM.

CONCEPT: Tool Composition
  These tools can be composed by the Ticket Agent:
    1. search_tickets → find relevant existing tickets
    2. create_ticket  → create a new one if not duplicate
    3. update_ticket  → change status/resolution

  The agent decides the ORDER and COMBINATION based on the request.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Input Schemas ────────────────────────────────────────────────────────────

class TicketSearchInput(BaseModel):
    """Input schema for searching tickets."""
    query: str = Field(
        description=(
            "Search query to find tickets. Can match title, description, or category. "
            "Examples: 'Redis connection timeout', 'Kubernetes pod crash', 'database backup'"
        )
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: 'open', 'in_progress', 'resolved', 'closed'. Leave null for all.",
    )
    priority: Optional[str] = Field(
        default=None,
        description="Filter by priority: 'low', 'medium', 'high', 'critical'. Leave null for all.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of tickets to return.",
    )


class TicketCreateInput(BaseModel):
    """Input schema for creating a new ticket."""
    title: str = Field(
        min_length=5,
        max_length=200,
        description="Short, descriptive title for the ticket. Be specific.",
    )
    description: str = Field(
        min_length=10,
        description="Detailed description of the issue, including symptoms, impact, and context.",
    )
    priority: str = Field(
        description="Ticket priority: 'low', 'medium', 'high', or 'critical'.",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Category of the issue. Examples: 'database', 'kubernetes', "
            "'networking', 'security', 'performance', 'deployment'."
        ),
    )


class TicketUpdateInput(BaseModel):
    """Input schema for updating an existing ticket."""
    ticket_id: int = Field(
        description="The numeric ID of the ticket to update (e.g., 42).",
    )
    status: Optional[str] = Field(
        default=None,
        description="New status: 'open', 'in_progress', 'resolved', 'closed'.",
    )
    priority: Optional[str] = Field(
        default=None,
        description="New priority: 'low', 'medium', 'high', 'critical'.",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Resolution notes. Required when setting status to 'resolved'.",
    )


class TicketGetInput(BaseModel):
    """Input schema for getting a specific ticket by ID."""
    ticket_id: int = Field(
        description="The numeric ID of the ticket to retrieve.",
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_ticket_tools(db) -> list[StructuredTool]:
    """
    Factory that returns all ticket-related tools with a bound DB session.

    Returns a list of tools:
      - search_tickets:  Find tickets by text + filters
      - create_ticket:   Create a new support ticket
      - update_ticket:   Update status/priority/resolution
      - get_ticket:      Get full details of a specific ticket

    Args:
        db: AsyncSession for database access

    Returns:
        List of configured StructuredTools for the Ticket Agent.
    """

    # ── Tool 1: Search Tickets ─────────────────────────────────────────────────
    async def search_tickets(
        query: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Search for existing support tickets by text and optional filters.

        Use this tool BEFORE creating a ticket to check for duplicates.
        Also use it when users ask "what tickets are open?" or
        "show me high-priority tickets".

        Returns a formatted list of matching tickets with their IDs, titles,
        status, and priority.
        """
        from sqlalchemy import select, or_
        from app.db.models import Ticket

        logger.info("ticket_search_tool", query=query[:80], status=status, priority=priority)

        try:
            stmt = select(Ticket).order_by(Ticket.created_at.desc())

            # Text search across title + description
            if query.strip():
                stmt = stmt.where(
                    or_(
                        Ticket.title.ilike(f"%{query}%"),
                        Ticket.description.ilike(f"%{query}%"),
                        Ticket.category.ilike(f"%{query}%"),
                    )
                )

            # Apply optional filters
            if status:
                stmt = stmt.where(Ticket.status == status.lower())
            if priority:
                stmt = stmt.where(Ticket.priority == priority.lower())

            stmt = stmt.limit(limit)
            result = await db.execute(stmt)
            tickets = result.scalars().all()

            if not tickets:
                return f"No tickets found matching '{query}'" + (
                    f" with status='{status}'" if status else ""
                ) + (f" and priority='{priority}'" if priority else "") + "."

            lines = [f"Found {len(tickets)} ticket(s):\n"]
            for t in tickets:
                lines.append(
                    f"• TKT-{t.id}: [{t.priority.upper()}] {t.title}\n"
                    f"  Status: {t.status} | Category: {t.category or 'N/A'}\n"
                    f"  {t.description[:120]}{'...' if len(t.description) > 120 else ''}"
                )
            return "\n".join(lines)

        except Exception as e:
            logger.error("ticket_search_failed", error=str(e))
            return f"Ticket search failed: {str(e)}"

    # ── Tool 2: Create Ticket ──────────────────────────────────────────────────
    async def create_ticket(
        title: str,
        description: str,
        priority: str,
        category: Optional[str] = None,
    ) -> str:
        """
        Create a new support ticket in the system.

        Use this when the user reports a new issue that needs to be tracked.
        Always search for existing tickets first to avoid duplicates.

        Returns the new ticket ID and confirmation message.
        """
        from app.db.models import Ticket, TicketPriority, TicketStatus

        logger.info("ticket_create_tool", title=title[:80], priority=priority)

        # Validate priority
        valid_priorities = [p.value for p in TicketPriority]
        if priority.lower() not in valid_priorities:
            return f"Invalid priority '{priority}'. Valid values: {', '.join(valid_priorities)}"

        try:
            ticket = Ticket(
                title=title,
                description=description,
                priority=priority.lower(),
                status=TicketStatus.OPEN,
                category=category,
                created_by=1,   # Placeholder until Phase 9 auth
            )
            db.add(ticket)
            await db.flush()   # Get the auto-generated ID
            await db.commit()

            logger.info("ticket_created_by_agent", ticket_id=ticket.id, title=title[:80])

            return (
                f"✅ Ticket TKT-{ticket.id} created successfully!\n"
                f"  Title:    {ticket.title}\n"
                f"  Priority: {ticket.priority.upper()}\n"
                f"  Status:   {ticket.status}\n"
                f"  Category: {ticket.category or 'N/A'}\n\n"
                f"The ticket has been logged and is now open for assignment."
            )

        except Exception as e:
            await db.rollback()
            logger.error("ticket_create_failed", error=str(e))
            return f"Failed to create ticket: {str(e)}"

    # ── Tool 3: Update Ticket ──────────────────────────────────────────────────
    async def update_ticket(
        ticket_id: int,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> str:
        """
        Update the status, priority, or resolution of an existing ticket.

        Use this when:
        - An issue has been resolved (set status='resolved', add resolution)
        - Priority needs to change due to new information
        - Ticket is being worked on (set status='in_progress')

        Returns confirmation of what was changed.
        """
        from sqlalchemy import select
        from app.db.models import Ticket
        from datetime import datetime, timezone

        logger.info("ticket_update_tool", ticket_id=ticket_id, status=status)

        try:
            result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
            ticket = result.scalar_one_or_none()

            if not ticket:
                return f"Ticket TKT-{ticket_id} not found. Please verify the ticket ID."

            changes = []
            if status:
                old_status = ticket.status
                ticket.status = status.lower()
                changes.append(f"status: {old_status} → {status.lower()}")
                if status.lower() == "resolved":
                    ticket.resolved_at = datetime.now(timezone.utc)

            if priority:
                old_priority = ticket.priority
                ticket.priority = priority.lower()
                changes.append(f"priority: {old_priority} → {priority.lower()}")

            if resolution:
                ticket.resolution = resolution
                changes.append("resolution: added")

            if not changes:
                return f"No changes specified for TKT-{ticket_id}. Provide at least one field to update."

            await db.commit()
            logger.info("ticket_updated_by_agent", ticket_id=ticket_id, changes=changes)

            return (
                f"✅ Ticket TKT-{ticket_id} updated successfully!\n"
                f"  Changes: {', '.join(changes)}\n"
                f"  Current status: {ticket.status}\n"
                f"  Current priority: {ticket.priority.upper()}"
            )

        except Exception as e:
            await db.rollback()
            logger.error("ticket_update_failed", ticket_id=ticket_id, error=str(e))
            return f"Failed to update ticket TKT-{ticket_id}: {str(e)}"

    # ── Tool 4: Get Ticket Details ─────────────────────────────────────────────
    async def get_ticket(ticket_id: int) -> str:
        """
        Get the full details of a specific ticket by its ID.

        Use this when the user asks "what's the status of TKT-42?" or
        when you need the full ticket details before making a decision.

        Returns all fields including title, description, status, resolution, timestamps.
        """
        from sqlalchemy import select
        from app.db.models import Ticket

        logger.info("ticket_get_tool", ticket_id=ticket_id)

        try:
            result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
            ticket = result.scalar_one_or_none()

            if not ticket:
                return f"Ticket TKT-{ticket_id} not found."

            return (
                f"Ticket TKT-{ticket_id} Details:\n"
                f"  Title:       {ticket.title}\n"
                f"  Status:      {ticket.status}\n"
                f"  Priority:    {ticket.priority.upper()}\n"
                f"  Category:    {ticket.category or 'N/A'}\n"
                f"  Description: {ticket.description}\n"
                f"  Created:     {ticket.created_at}\n"
                f"  Updated:     {ticket.updated_at}\n"
                + (f"  Resolution:  {ticket.resolution}\n" if ticket.resolution else "")
                + (f"  Resolved At: {ticket.resolved_at}\n" if ticket.resolved_at else "")
            )

        except Exception as e:
            logger.error("ticket_get_failed", ticket_id=ticket_id, error=str(e))
            return f"Failed to retrieve ticket TKT-{ticket_id}: {str(e)}"

    # ── Build and return all tools ─────────────────────────────────────────────
    return [
        StructuredTool.from_function(
            coroutine=search_tickets,
            name="search_tickets",
            description=(
                "Search for existing support tickets by keyword, status, and priority. "
                "Always use this BEFORE creating a new ticket to avoid duplicates. "
                "Also use to answer questions like 'show me open tickets' or 'find Redis issues'."
            ),
            args_schema=TicketSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=create_ticket,
            name="create_ticket",
            description=(
                "Create a new support ticket. Extract title, description, priority, "
                "and category from the user's request. "
                "Search for existing tickets first to avoid duplicates."
            ),
            args_schema=TicketCreateInput,
        ),
        StructuredTool.from_function(
            coroutine=update_ticket,
            name="update_ticket",
            description=(
                "Update an existing ticket's status, priority, or resolution. "
                "Use when resolving issues, changing priority, or marking as in-progress. "
                "Requires the numeric ticket ID (e.g., 42 for TKT-42)."
            ),
            args_schema=TicketUpdateInput,
        ),
        StructuredTool.from_function(
            coroutine=get_ticket,
            name="get_ticket",
            description=(
                "Get full details of a specific ticket by its numeric ID. "
                "Use when the user asks about a specific ticket number."
            ),
            args_schema=TicketGetInput,
        ),
    ]

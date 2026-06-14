"""
agents/rca_agent.py — Root Cause Analysis Sub-Agent
=====================================================
CONCEPT: Sub-Agents (Agent Composition)

  A sub-agent is called BY another agent (not directly by the user).
  The Incident Agent gathers evidence and then calls the RCA Agent
  as a "specialist consultant" to analyze that evidence.

  AGENT COMPOSITION PATTERN:
    User → Supervisor → Incident Agent → RCA Agent (sub-agent)
                                      → other tools

  Why use a sub-agent instead of one big prompt?
    1. SEPARATION OF CONCERNS: Incident Agent gathers; RCA Agent analyzes
    2. FOCUSED CONTEXT: RCA Agent sees only the evidence, not all chat history
    3. SPECIALIZED PROMPT: RCA Agent's prompt is tuned for structured analysis
    4. REUSABILITY: RCA Agent can be called from other agents too

CONCEPT: RCA Methodology (ITL/SRE)
  Root Cause Analysis follows structured methodologies:

  5 WHYS technique:
    Why is the payment service down?
    → Because pods are OOMKilled
    Why are pods OOMKilled?
    → Because memory usage is 98%
    Why is memory at 98%?
    → Because there's a memory leak
    Why is there a memory leak?
    → Because v2.5 introduced unbounded connection pooling
    Why was unbounded pooling introduced?
    → Because load testing didn't simulate sustained traffic

  Fishbone (Ishikawa) categories:
    - Environment (infrastructure, cloud, network)
    - Process (deployment, change management)
    - Tools (software versions, dependencies)
    - People (human error, misconfiguration)

  The RCA Agent applies these frameworks to the evidence provided.

CONCEPT: Evidence-Based Analysis
  The RCA Agent operates ONLY on evidence provided to it.
  It does NOT search for more data itself — that's the Incident Agent's job.
  This separation keeps the analysis phase clean and reproducible.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────

RCA_AGENT_SYSTEM_PROMPT = """You are a Root Cause Analysis (RCA) specialist for {app_name}.

You receive structured evidence from the Incident Agent and your job is to determine:
1. The ROOT CAUSE of the incident (the underlying technical reason)
2. CONTRIBUTING FACTORS (things that made it worse)
3. RECOMMENDED REMEDIATION (immediate fix + long-term prevention)

## Your Analysis Framework

### Step 1: Timeline Reconstruction
- Identify WHEN things started going wrong based on the evidence
- Find the FIRST failing component — this is often the root cause

### Step 2: Pattern Recognition
- Look for resource exhaustion (OOM, disk full, max connections)
- Look for cascading failures (A failed → B timed out → C errored)
- Look for deployment correlation (did a deployment happen before the issue?)
- Look for similar historical incidents (same pattern before?)

### Step 3: Root Cause Determination
Apply the 5 Whys technique:
  - Why did X fail? → Because Y
  - Why did Y happen? → Because Z
  - Continue until you reach the technical root cause

### Step 4: Recommendations
Always provide:
  - IMMEDIATE action (fix right now)
  - SHORT-TERM action (prevent recurrence this week)
  - LONG-TERM action (systemic improvement)

## Required Output Format

Your response MUST follow this structure:

## 🔍 Root Cause Analysis

### Incident Summary
[Brief summary of what happened]

### Timeline
[Key events in chronological order based on evidence]

### Root Cause
**[Technical root cause in one sentence]**

[Detailed explanation of why this happened]

### Contributing Factors
- [Factor 1]
- [Factor 2]

### Recommended Remediation

**Immediate (do now):**
- [Action 1]
- [Action 2]

**Short-term (this week):**
- [Action 1]

**Long-term (systemic):**
- [Action 1]

### Confidence Level
[High/Medium/Low] — [Explanation of certainty]

Current Platform: {app_name} v{app_version}
"""


async def run_rca_analysis(evidence: dict) -> str:
    """
    Run Root Cause Analysis on the provided evidence.

    CONCEPT: Direct LLM Call (not a full agent)
      RCA doesn't need a tool-calling ReAct loop — it's a single,
      focused analysis task. We call the LLM directly with the evidence
      as context and get a structured response.

      Using a full ReAct agent would be overkill here. The LLM has all
      the information it needs in the evidence dict — no tool calls required.

    CONCEPT: Evidence Structure
      The evidence dict contains:
        - logs:             Raw log entries from logs_tool
        - metrics:          Metric values from metrics_tool
        - incident_history: Similar past incidents
        - runbook_context:  Relevant documentation from RAG
        - incident_details: The incident being investigated

    Args:
        evidence: Dict containing all gathered evidence

    Returns:
        Formatted RCA report as a string.
    """
    logger.info(
        "rca_analysis_started",
        evidence_keys=list(evidence.keys()),
        has_logs=bool(evidence.get("logs")),
        has_metrics=bool(evidence.get("metrics")),
    )

    # ── Build LLM ─────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,     # Slightly creative for analysis, but mostly deterministic
        api_key=settings.openai_api_key,
        max_tokens=settings.agent_max_tokens,
    )

    # ── Format Evidence for the LLM ───────────────────────────────────────────
    # CONCEPT: Context Window Management
    #   We must fit all evidence within the LLM's context window.
    #   We truncate each section to avoid exceeding token limits.
    #   Priority: incident_details > logs > metrics > history > runbook

    max_section_length = 1500  # Characters per evidence section

    evidence_text = "## Evidence for Analysis\n\n"

    if evidence.get("incident_details"):
        evidence_text += f"### Incident Details\n{str(evidence['incident_details'])[:max_section_length]}\n\n"

    if evidence.get("logs"):
        evidence_text += f"### Application Logs\n{str(evidence['logs'])[:max_section_length]}\n\n"

    if evidence.get("metrics"):
        evidence_text += f"### System Metrics\n{str(evidence['metrics'])[:max_section_length]}\n\n"

    if evidence.get("incident_history"):
        evidence_text += f"### Similar Past Incidents\n{str(evidence['incident_history'])[:max_section_length]}\n\n"

    if evidence.get("runbook_context"):
        evidence_text += f"### Relevant Documentation\n{str(evidence['runbook_context'])[:max_section_length]}\n\n"

    # ── Construct Messages ─────────────────────────────────────────────────────
    system_prompt = RCA_AGENT_SYSTEM_PROMPT.format(
        app_name=settings.app_name,
        app_version=settings.app_version,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Please analyze the following evidence and provide a complete Root Cause Analysis.\n\n"
            f"{evidence_text}"
        )),
    ]

    try:
        # ── Direct LLM Call ────────────────────────────────────────────────────
        response = await llm.ainvoke(messages)
        rca_result = response.content

        logger.info(
            "rca_analysis_complete",
            result_length=len(rca_result),
        )

        return rca_result

    except Exception as e:
        logger.error("rca_analysis_failed", error=str(e), exc_info=True)
        return (
            "## Root Cause Analysis\n\n"
            "⚠️ Unable to complete RCA due to an error.\n\n"
            f"**Error**: {str(e)}\n\n"
            "Please review the evidence manually and consult the on-call runbook."
        )

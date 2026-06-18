import re
import os
from pathlib import Path
from pydantic import BaseModel
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Try to import NeMo Guardrails defensively to prevent startup crashes if compilation failed
try:
    from nemoguardrails import LLMRails, RailsConfig
    NEMO_AVAILABLE = True
except ImportError as e:
    logger.warning("nemo_guardrails_not_available", error=str(e))
    NEMO_AVAILABLE = False

# Regex patterns for sensitive data detection (API Keys, private keys, passwords)
SENSITIVE_PATTERNS = [
    r"(?i)password\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)api[-_]?key\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)secret[-_]?key\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+",
    r"(?i)gh[op]_[a-zA-Z0-9]{36}",  # GitHub personal access tokens
    r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----",  # Private keys
    r"(?i)aws[_-]?access[_-]?key[_-]?id\s*=\s*['\"][A-Z0-9]{20}['\"]",
    r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*=\s*['\"][a-zA-Z0-9/+=]{40}['\"]"
]

REFUSAL_MESSAGE = (
    "I cannot process this request because it violates safety guidelines, "
    "contains inappropriate content, or is off-topic for this SRE assistant."
)

SENSITIVE_DATA_BLOCKED_MESSAGE = (
    "I cannot process this request because it contains sensitive information "
    "(such as passwords, API keys, or private tokens) that should not be transmitted."
)

class GuardrailsResult(BaseModel):
    """Result of the guardrails check."""
    blocked: bool
    message: str
    error_message: str | None = None


def contains_sensitive_data(text: str) -> bool:
    """Scan text using regular expressions to detect leaked secrets/credentials."""
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


class InputGuardrailsRunner:
    """
    Singleton wrapper for NeMo Guardrails LLMRails instance.
    Initializes once on startup to avoid overhead during chat requests.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InputGuardrailsRunner, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.rails = None
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        if not settings.enable_guardrails:
            logger.info("guardrails_disabled_by_settings")
            self.initialized = True
            self.rails = None
            return

        if not NEMO_AVAILABLE:
            logger.error(
                "nemo_guardrails_initialization_skipped",
                reason="nemoguardrails package is not installed or failed to import."
            )
            self.initialized = True
            self.rails = None
            return

        try:
            # Resolve config folder path
            # BASE_DIR is backend/ app/parent directory
            from app.config import BASE_DIR
            config_path = Path(BASE_DIR) / settings.guardrails_config_path

            logger.info("loading_nemo_guardrails_config", path=str(config_path))
            if not config_path.exists():
                logger.error("nemo_guardrails_config_path_not_found", path=str(config_path))
                self.rails = None
            else:
                config = RailsConfig.from_path(str(config_path))
                self.rails = LLMRails(config)
                logger.info("nemo_guardrails_initialized_successfully")
        except Exception as e:
            logger.error("nemo_guardrails_initialization_failed", error=str(e), exc_info=True)
            self.rails = None

        self.initialized = True

    async def check(self, message: str) -> GuardrailsResult:
        """
        Check user input message against all active rails.
        """
        # 1. First run the lightweight regex sensitive data filter (no LLM latency)
        if contains_sensitive_data(message):
            logger.warning("guardrails_blocked_sensitive_data")
            return GuardrailsResult(
                blocked=True,
                message=SENSITIVE_DATA_BLOCKED_MESSAGE
            )

        # 2. Check if Guardrails is enabled and initialized
        if not settings.enable_guardrails or not NEMO_AVAILABLE:
            return GuardrailsResult(blocked=False, message="")

        self.initialize()

        if not self.rails:
            # If NeMo Guardrails failed to load but is enabled, fail open or log warning
            logger.warning("guardrails_runner_not_initialized_falling_back_to_pass_through")
            return GuardrailsResult(blocked=False, message="")

        try:
            # Format message for NeMo Guardrails API
            history = [{"role": "user", "content": message}]
            
            logger.info("executing_nemo_guardrails_check", message_preview=message[:80])
            
            # NeMo Guardrails generates response asynchronously
            response = await self.rails.generate_async(messages=history)
            
            # Extract content from various potential response formats
            content = ""
            if isinstance(response, dict):
                content = response.get("content", "")
            elif isinstance(response, list) and len(response) > 0:
                last_msg = response[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                else:
                    content = getattr(last_msg, "content", str(last_msg))
            else:
                content = str(response)

            content = content.strip()

            # Check if response matches our custom refusal or standard default refusals
            default_refusals = [
                REFUSAL_MESSAGE,
                "I'm sorry, I can't respond to that.",
                "I'm sorry, I can't help with that.",
                "I cannot answer this question."
            ]

            is_blocked = False
            for refusal in default_refusals:
                if refusal.lower() in content.lower():
                    is_blocked = True
                    break

            if is_blocked:
                logger.warning("guardrails_blocked_input", refusal_received=content[:100])
                return GuardrailsResult(
                    blocked=True,
                    message=REFUSAL_MESSAGE
                )

            logger.info("guardrails_passed_input")
            return GuardrailsResult(blocked=False, message="")

        except Exception as e:
            logger.error("guardrails_execution_failed", error=str(e), exc_info=True)
            # Fail open for safety in production but log the error
            return GuardrailsResult(
                blocked=False,
                message="",
                error_message=f"Guardrails execution error: {str(e)}"
            )


# Initialize the singleton runner
guardrails_runner = InputGuardrailsRunner()

async def check_input_guardrails(message: str) -> GuardrailsResult:
    """
    Public API helper function to run input guardrails.
    """
    return await guardrails_runner.check(message)

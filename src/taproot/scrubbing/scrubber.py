from __future__ import annotations

import copy
import re

_PATTERNS: list[tuple[str, str]] = [
    # URLs first (broader — catches http/https/ftp URLs before hostname pattern)
    ("URL", r"https?://[^\s\"'<>]+|ftp://[^\s\"'<>]+"),
    # IPv6
    ("IP", r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
    # IPv4
    ("IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Email
    ("EMAIL", r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    # Hostnames: word(s) with dots and common internal suffixes
    ("HOST", r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)+\b"),
    # Name-like patterns: two consecutive title-case words
    # Skipped if they match known service/system words
    ("NAME", r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b"),
]

# Words that look like names but are not PII — suppress NAME scrubbing for these
_NAME_ALLOWLIST: set[str] = {
    "New User", "End User", "Test User", "Platform Engineering", "Network Operations",
    "Service Desk", "Change Management", "Problem Management", "Incident Management",
    "Root Cause", "Work Around", "High Priority", "Medium Priority", "Low Priority",
    "Problem Record", "Stack Trace", "Auth Service", "Email Service",
    "Azure Open", "Open AI", "Aws Bedrock",
}

# HTTP status codes and error codes should NOT be scrubbed by the IP pattern
_STATUS_CODE_RE = re.compile(r"HTTP\s*\d{3}|ERR_[A-Z0-9_]+|\b\d{3}\s")


class DataScrubber:
    """
    Local PII and sensitive data scrubber.
    Runs entirely on-device. No data leaves for scrubbing.

    Scrubs: emails, IPs, hostnames, URLs, name-like patterns.
    Does NOT scrub: error codes, HTTP status codes, service names, timestamps.
    """

    def scrub_text(
        self,
        text: str,
        _shared_mapping: dict[str, str] | None = None,
        _shared_counters: dict[str, int] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """
        Anonymise sensitive entities in text.
        Returns: (scrubbed_text, mapping) where mapping is {placeholder → original}.
        Pass _shared_mapping/_shared_counters to accumulate state across multiple calls.
        """
        mapping: dict[str, str] = _shared_mapping if _shared_mapping is not None else {}
        counters: dict[str, int] = _shared_counters if _shared_counters is not None else {}
        result = text

        for entity_type, pattern in _PATTERNS:
            regex = re.compile(pattern)

            def replace_match(m: re.Match, et: str = entity_type) -> str:
                original = m.group(0)
                if et == "NAME" and original in _NAME_ALLOWLIST:
                    return original
                if original.startswith("<<") and original.endswith(">>"):
                    return original
                # Reuse existing placeholder for the same value
                for ph, orig in mapping.items():
                    if orig == original:
                        return ph
                counters[et] = counters.get(et, 0) + 1
                placeholder = f"<<{et}_{counters[et]}>>"
                mapping[placeholder] = original
                return placeholder

            result = regex.sub(replace_match, result)

        return result, mapping

    def restore_text(self, text: str, mapping: dict[str, str]) -> str:
        """Replace placeholders in text with original values."""
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    def scrub_messages(self, messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
        """Scrub all content fields in a message list. Returns scrubbed messages + combined mapping."""
        combined_mapping: dict[str, str] = {}
        combined_counters: dict[str, int] = {}
        scrubbed = []
        for msg in messages:
            scrubbed_msg = copy.deepcopy(msg)
            content = scrubbed_msg.get("content", "")
            if isinstance(content, str):
                scrubbed_content, _ = self.scrub_text(
                    content, _shared_mapping=combined_mapping, _shared_counters=combined_counters
                )
                scrubbed_msg["content"] = scrubbed_content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        for field in ("text", "content"):
                            if field in block and isinstance(block[field], str):
                                scrubbed_val, _ = self.scrub_text(
                                    block[field],
                                    _shared_mapping=combined_mapping,
                                    _shared_counters=combined_counters,
                                )
                                block[field] = scrubbed_val
            scrubbed.append(scrubbed_msg)
        return scrubbed, combined_mapping

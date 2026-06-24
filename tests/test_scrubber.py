import pytest

from taproot.scrubbing.scrubber import DataScrubber


@pytest.fixture
def scrubber() -> DataScrubber:
    return DataScrubber()


class TestScrubText:
    def test_scrubs_email_addresses(self, scrubber: DataScrubber) -> None:
        text, mapping = scrubber.scrub_text("Contact john.doe@example.com for details.")
        assert "john.doe@example.com" not in text
        assert "<<EMAIL_1>>" in text
        assert mapping["<<EMAIL_1>>"] == "john.doe@example.com"

    def test_scrubs_ipv4_addresses(self, scrubber: DataScrubber) -> None:
        text, mapping = scrubber.scrub_text("Server at 192.168.1.100 is unresponsive.")
        assert "192.168.1.100" not in text
        assert any("<<IP_" in k for k in mapping)

    def test_scrubs_urls(self, scrubber: DataScrubber) -> None:
        text, mapping = scrubber.scrub_text("See https://internal.corp.com/docs for details.")
        assert "https://internal.corp.com/docs" not in text
        assert any("<<URL_" in k for k in mapping)

    def test_does_not_scrub_error_codes(self, scrubber: DataScrubber) -> None:
        text, _ = scrubber.scrub_text("HTTP 503 service unavailable. ERR_AUTH_001 raised.")
        # Error codes must not be replaced
        assert "503" in text
        assert "ERR_AUTH_001" in text

    def test_multiple_emails_get_distinct_placeholders(self, scrubber: DataScrubber) -> None:
        text, mapping = scrubber.scrub_text(
            "Contact alice@corp.com or bob@corp.com for help."
        )
        assert len([k for k in mapping if k.startswith("<<EMAIL_")]) == 2
        assert "alice@corp.com" not in text
        assert "bob@corp.com" not in text

    def test_same_value_gets_same_placeholder(self, scrubber: DataScrubber) -> None:
        text, mapping = scrubber.scrub_text(
            "alice@corp.com reported the issue. Reply to alice@corp.com."
        )
        email_placeholders = [k for k in mapping if k.startswith("<<EMAIL_")]
        assert len(email_placeholders) == 1
        # Both occurrences replaced by the same placeholder
        assert text.count("<<EMAIL_1>>") == 2


class TestRestoreText:
    def test_round_trip(self, scrubber: DataScrubber) -> None:
        original = "User john.doe@example.com logged in from 10.0.0.5."
        scrubbed, mapping = scrubber.scrub_text(original)
        restored = scrubber.restore_text(scrubbed, mapping)
        assert restored == original

    def test_restore_no_op_when_no_placeholders(self, scrubber: DataScrubber) -> None:
        text = "No PII here, just normal text."
        assert scrubber.restore_text(text, {}) == text

    def test_restore_multiple_placeholders(self, scrubber: DataScrubber) -> None:
        original = "Email alice@a.com, host prod-01.internal, IP 10.0.0.1."
        scrubbed, mapping = scrubber.scrub_text(original)
        restored = scrubber.restore_text(scrubbed, mapping)
        assert restored == original


class TestScrubMessages:
    def test_scrubs_user_message_content(self, scrubber: DataScrubber) -> None:
        messages = [
            {"role": "user", "content": "Ticket from bob@corp.com about 10.0.0.1."},
        ]
        scrubbed, mapping = scrubber.scrub_messages(messages)
        assert "bob@corp.com" not in scrubbed[0]["content"]
        assert len(mapping) > 0

    def test_scrubs_list_content_blocks(self, scrubber: DataScrubber) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "admin@corp.com reported it."},
                ],
            }
        ]
        scrubbed, mapping = scrubber.scrub_messages(messages)
        assert "admin@corp.com" not in scrubbed[0]["content"][0]["content"]

    def test_returns_combined_mapping(self, scrubber: DataScrubber) -> None:
        messages = [
            {"role": "user", "content": "alice@a.com"},
            {"role": "assistant", "content": "bob@b.com"},
        ]
        _, mapping = scrubber.scrub_messages(messages)
        emails = [v for v in mapping.values() if "@" in v]
        assert len(emails) == 2

    def test_original_messages_not_mutated(self, scrubber: DataScrubber) -> None:
        messages = [{"role": "user", "content": "alice@corp.com is affected."}]
        scrubber.scrub_messages(messages)
        assert messages[0]["content"] == "alice@corp.com is affected."

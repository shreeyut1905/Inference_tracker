from __future__ import annotations

import html
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any
from zoneinfo import ZoneInfo

from .config import Settings
from .models import Paper
from .time_utils import as_utc


class EmailError(RuntimeError):
    pass


class EmailSender:
    def __init__(self, settings: Settings, smtp_factory: Any | None = None) -> None:
        self.settings = settings
        self.smtp_factory = smtp_factory or self._default_factory

    def send(self, papers: list[Paper], now: datetime) -> None:
        if not self.settings.mail_to or not self.settings.mail_from:
            raise EmailError("Email recipient and sender are not configured")
        message = self.build_message(papers, now)
        smtp = self.smtp_factory(self.settings)
        try:
            if self.settings.smtp_username and self.settings.smtp_password:
                smtp.login(self.settings.smtp_username, self.settings.smtp_password)
            smtp.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise EmailError(f"SMTP delivery failed: {error}") from error
        finally:
            try:
                smtp.quit()
            except (OSError, smtplib.SMTPException):
                pass

    def build_message(self, papers: list[Paper], now: datetime) -> EmailMessage:
        local_now = as_utc(now).astimezone(ZoneInfo(self.settings.timezone))
        date_text = local_now.strftime("%Y-%m-%d")
        subject = f"Diffusion efficiency papers — {len(papers)} — {date_text}"
        plain = self._render_plain(papers, date_text)
        html_body = self._render_html(papers, date_text)
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.mail_from or ""
        message["To"] = ", ".join(self.settings.mail_to)
        message.set_content(plain)
        message.add_alternative(html_body, subtype="html")
        return message

    def _default_factory(self, settings: Settings) -> smtplib.SMTP:
        if not settings.smtp_host:
            raise EmailError("SMTP_HOST is not configured")
        if settings.smtp_security == "ssl":
            return smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30)
        client = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
        if settings.smtp_security == "starttls":
            client.ehlo()
            client.starttls()
            client.ehlo()
        return client

    @staticmethod
    def _render_plain(papers: list[Paper], date_text: str) -> str:
        lines = [f"Diffusion efficiency papers for {date_text}", ""]
        if not papers:
            lines.append("No new useful papers matched the tracker criteria.")
            return "\n".join(lines)
        for section_name, section_papers in _source_sections(papers):
            lines.extend([section_name, "-" * len(section_name)])
            if not section_papers:
                lines.extend(["No new papers from this source.", ""])
                continue
            for index, paper in enumerate(section_papers, start=1):
                lines.extend(
                    [
                        f"{index}. {paper.title}",
                        f"Authors: {', '.join(paper.authors) or 'Unknown'}",
                        f"Category: {paper.useful_category}",
                        f"Confidence: {paper.useful_confidence:.2f}",
                        f"Assessment: {paper.useful_reason or 'No assessment available.'}",
                        f"Abstract: {_shorten(paper.abstract)}",
                        f"Links: {_plain_links(paper)}",
                        "",
                    ]
                )
        return "\n".join(lines)

    @staticmethod
    def _render_html(papers: list[Paper], date_text: str) -> str:
        escaped_date = html.escape(date_text)
        parts = [
            "<html><body style=\"font-family:Arial,sans-serif;line-height:1.5\">",
            f"<h1>Diffusion efficiency papers for {escaped_date}</h1>",
        ]
        if not papers:
            parts.append("<p>No new useful papers matched the tracker criteria.</p>")
        for section_name, section_papers in _source_sections(papers):
            source = "arxiv" if section_name == "arXiv papers" else "huggingface"
            parts.append(f"<h2>{html.escape(section_name)}</h2>")
            if not section_papers:
                parts.append("<p>No new papers from this source.</p>")
                continue
            for paper in section_papers:
                title_url = _source_url(paper, source)
                title_link = _html_link(title_url, paper.title)
                parts.extend(
                    [
                        "<article style=\"border-top:1px solid #ddd;padding:12px 0\">",
                        f"<h3>{title_link}</h3>",
                        f"<p><strong>Authors:</strong> "
                        f"{html.escape(', '.join(paper.authors) or 'Unknown')}</p>",
                        f"<p><strong>Category:</strong> {html.escape(paper.useful_category)} "
                        f"<strong>Confidence:</strong> {paper.useful_confidence:.2f}</p>",
                        f"<p>{html.escape(paper.useful_reason or 'No assessment available.')}</p>",
                        f"<p>{html.escape(_shorten(paper.abstract))}</p>",
                        f"<p>{_html_link_list(paper)}</p>",
                        "</article>",
                    ]
                )
        parts.append("</body></html>")
        return "".join(parts)


def _shorten(value: str, limit: int = 700) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _source_sections(papers: list[Paper]) -> tuple[tuple[str, list[Paper]], ...]:
    return (
        ("arXiv papers", [paper for paper in papers if _belongs_to_source(paper, "arxiv")]),
        (
            "Hugging Face Hub papers",
            [paper for paper in papers if _belongs_to_source(paper, "huggingface")],
        ),
    )


def _belongs_to_source(paper: Paper, source: str) -> bool:
    return source in paper.sources or source in paper.urls


def _source_url(paper: Paper, source: str) -> str:
    return paper.urls.get(source, "")


def _plain_links(paper: Paper) -> str:
    labels = {
        "arxiv": "arXiv paper",
        "huggingface": "Hugging Face Hub",
        "pdf": "PDF",
    }
    links = [
        f"{labels.get(name, name)}: {url}"
        for name, url in paper.urls.items()
        if url
    ]
    if paper.github_url:
        links.append(f"Code: {paper.github_url}")
    return ", ".join(links) or "No links available."


def _html_link(url: str, label: str) -> str:
    if not url:
        return html.escape(label)
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def _html_link_list(paper: Paper) -> str:
    labels = {
        "arxiv": "arXiv paper",
        "huggingface": "Hugging Face Hub",
        "pdf": "PDF",
    }
    links = [
        _html_link(url, labels.get(name, name))
        for name, url in paper.urls.items()
        if url
    ]
    if paper.github_url:
        links.append(_html_link(paper.github_url, "Code"))
    return " · ".join(links) or "No links available."

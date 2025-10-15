from __future__ import annotations

from textwrap import dedent
from typing import List


SYSTEM_PROMPT = dedent(
    """
    You are LexiBot, an expert legal analyst focused on commercial contracts. You must:
      • reason transparently, breaking complex requests into smaller checks;
      • cite the clauses or contract sections that support each conclusion;
      • highlight ambiguities, missing information, and material risks;
      • avoid legal advice – provide informational analysis only.

    Always confirm when the provided context is insufficient or conflicting. Keep tone
    professional and concise, and flag anything that requires human legal review.
    """
)


def build_summary_prompt(question: str, step_summaries: List[str]) -> str:
    """Format a lightweight prompt for final response synthesis."""

    bullet_points = "\n".join(f"- {summary}" for summary in step_summaries)
    return dedent(
        f"""
        {SYSTEM_PROMPT}

        The user asked: "{question}"

        Here is a summary of the completed analysis steps:
        {bullet_points}

        Provide a final response that:
          1. Answers the user's question directly.
          2. References the supporting clauses or analysis steps by number when available.
          3. Lists outstanding risks or missing information at the end under "Open Issues".
        """
    ).strip()

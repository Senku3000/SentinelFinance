"""LLM-based structured data extraction from parsed financial documents.

Sends parsed document text to Groq to extract structured financial fields
(income, expenses, tax details, investments, etc.) that can be merged
into the user's profile.
"""

import json
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq

from ..config import Config


# Max characters to send per extraction call (~4000 tokens)
MAX_TEXT_LENGTH = 12000

EXTRACTION_PROMPT = """You are a financial document analyzer. Extract structured financial data from the following document text.

IMPORTANT RULES:
- Extract ONLY fields you can confidently determine from the document.
- Leave all uncertain fields as null.
- Do NOT guess or make assumptions.
- Monetary values should be numbers (not strings). Use INR (Indian Rupees) unless clearly stated otherwise.
- If the document contains monthly data, compute monthly averages where relevant.

Return a JSON object with this exact structure (set fields to null if not found):

{{
  "income": {{
    "monthly": <number or null>,
    "annual": <number or null>,
    "source": <string or null>
  }},
  "expenses": {{
    "monthly": <number or null>,
    "breakdown": {{
      "rent": <number or null>,
      "food": <number or null>,
      "transport": <number or null>,
      "utilities": <number or null>,
      "emi": <number or null>,
      "insurance": <number or null>,
      "other": <number or null>
    }}
  }},
  "tax_details": {{
    "hra_exemption": <number or null>,
    "section_80c": <number or null>,
    "section_80d": <number or null>,
    "other_deductions": <number or null>,
    "total_tax_paid": <number or null>,
    "taxable_income": <number or null>
  }},
  "existing_investments": {{
    "equity": <number or null>,
    "debt": <number or null>,
    "gold": <number or null>,
    "fd": <number or null>,
    "ppf": <number or null>,
    "nps": <number or null>,
    "others": <number or null>
  }},
  "liabilities": {{
    "home_loan_outstanding": <number or null>,
    "car_loan_outstanding": <number or null>,
    "personal_loan_outstanding": <number or null>,
    "credit_card_due": <number or null>
  }},
  "document_summary": "<one-line summary of what this document contains>"
}}

DOCUMENT TEXT:
{document_text}

Return ONLY the JSON object, no other text."""


class LLMExtractor:
    """Extracts structured financial data from document text using an LLM."""

    def __init__(self):
        self.llm = ChatGroq(
            model=Config.GROQ_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0.0,  # deterministic for extraction
        )

    def extract(self, document_text: str) -> Dict[str, Any]:
        """Extract structured financial data from document text.

        Args:
            document_text: Raw text parsed from a financial document

        Returns:
            Dictionary of extracted fields (null for fields not found)
        """
        if not document_text.strip():
            return {"error": "Empty document text"}

        # Truncate if too long
        text = document_text[:MAX_TEXT_LENGTH]
        if len(document_text) > MAX_TEXT_LENGTH:
            text += "\n\n[Document truncated — additional content not shown]"

        prompt = EXTRACTION_PROMPT.format(document_text=text)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Parse JSON from response
            extracted = self._parse_json(content)
            if extracted is None:
                return {"error": "Failed to parse LLM response", "raw": content}

            return extracted

        except Exception as e:
            return {"error": f"LLM extraction failed: {str(e)}"}

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code fences if present
        if "```" in text:
            lines = text.split("\n")
            json_lines = []
            inside = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
            return None


def merge_extracted_data(
    existing_profile: Dict[str, Any], extracted: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge LLM-extracted data into an existing user profile.

    Document-extracted data takes precedence — if the LLM confidently
    extracted a value from a real document, it should overwrite defaults
    and existing values.

    Args:
        existing_profile: Current user profile dict
        extracted: Extracted data from LLM

    Returns:
        Updated profile dict
    """
    if "error" in extracted:
        return existing_profile

    # Remove non-profile fields
    skip_keys = {"error", "raw", "document_summary"}

    for key, value in extracted.items():
        if key in skip_keys or value is None:
            continue

        if key not in existing_profile:
            existing_profile[key] = value
            continue

        existing_val = existing_profile[key]

        if isinstance(value, dict) and isinstance(existing_val, dict):
            # Merge nested dicts — extracted non-null values overwrite
            for sub_key, sub_val in value.items():
                if sub_val is None:
                    continue
                existing_val[sub_key] = sub_val
        else:
            existing_profile[key] = value

    return existing_profile

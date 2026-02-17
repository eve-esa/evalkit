"""
Common utilities for Multiple Choice Question Answering (MCQA) tasks.

This module provides shared functionality across all MCQA tasks including
label extraction, answer processing, and formatting.
"""

import re


def extract_label(answer: str) -> str:
    """
    Extract a single letter label from a model's answer.

    This function is designed to work with the lm-evaluation-harness regex filter.
    It extracts the first capital letter (A, B, C, D, etc.) from the answer,
    handling various formats including long reasoning followed by an answer.

    Handles formats like:
    - Simple letter: "A", "B"
    - Letter with period: "A.", "D."
    - Letter with text: "A. Some text here"
    - With prefix: "Answer: C"
    - Quoted format: "\"B\"", "'A'"
    - Parentheses format: "(C)", "(A)"
    - Bold markdown: "**B**", "**A**\n\nExplanation...", "**A. 4.65**", "**(B)**"
    - Italic markdown: "*B*", "*(A)*"
    - With reasoning: "...long reasoning... the answer is **(B)**."
    - Numbered list format with dash: "2. **Question?**\n   - B"
    - Answer-first CoT format: "B\n...long reasoning..."
    - Explain-then-answer CoT format: "...long reasoning...\nA"

    Args:
        answer: The model's answer string

    Returns:
        The extracted letter label (e.g., "A", "B", "C", "D")
        Returns empty string if no valid label is found.

    Examples:
        >>> extract_label("A")
        'A'
        >>> extract_label("Answer: B. Some text")
        'B'
        >>> extract_label("D.")
        'D'
        >>> extract_label("**B**")
        'B'
        >>> extract_label("**C**\\n\\nHere's why...")
        'C'
        >>> extract_label("(C)")
        'C'
        >>> extract_label("\"B\"")
        'B'
        >>> extract_label("...reasoning... the answer is **(B)**.")
        'B'
        >>> extract_label("2. **Question?**\\n   - B")
        'B'
    """
    # First, check for a bare letter on its own line at the very END of the text.
    # CoT models often reason at length and then output the answer alone on the last line.
    # E.g., "...long reasoning...\nA"
    # A letter is considered "bare" only if it is the sole non-whitespace content on that last line
    # (i.e., preceded by a newline or start-of-string, with only optional whitespace around it).
    bare_letter_end = re.search(r"(?:^|\n)\s*([A-Z])\s*$", answer)
    if bare_letter_end:
        return bare_letter_end.group(1)

    # Second, check for a bare letter on the very first line (answer-first CoT models).
    # E.g., "B\nThe question is... [long reasoning]"
    # This must run before dash_answer_matches which can be confused by bullet-point reasoning text.
    bare_letter_start = re.search(r"^\s*([A-Z])\s*\n", answer)
    if bare_letter_start:
        return bare_letter_start.group(1)

    # Third, try to find numbered list format with dash-prefixed answer
    # Pattern: optional numbering/bullets followed by question, then dash with letter
    # E.g., "2. **Question?**\n   - B"
    # Use findall to get all matches, then take the last one (most likely the actual question)
    # Use negative lookahead to avoid matching "- Option A" (letter followed by lowercase)
    dash_answer_matches = re.findall(
        r"(?:^|\n)\s*\d*\.?\s*\*{0,2}[^*\n]*\*{0,2}\s*\n\s*-\s*([A-Z])(?![a-z])",
        answer,
        flags=re.MULTILINE
    )
    if dash_answer_matches:
        return dash_answer_matches[-1]  # Return the last match

    # Second, try to find "answer is" pattern anywhere in the text
    # This handles: "the answer is (B)", "answer is **B**", etc.
    answer_is_match = re.search(
        r"(?:the\s+)?answer\s+is\s+[*_()]*([A-Z])[*_().\s]*", answer, flags=re.IGNORECASE
    )
    if answer_is_match:
        return answer_is_match.group(1)

    # Strip common prefixes at the beginning like "Answer: ", "answer: ", etc.
    answer = re.sub(
        r"^(?:Answer|answer|The answer is|the answer is):\s*", "", answer, flags=re.IGNORECASE
    )

    # First, explicitly check for bold markdown at the start: **A**, **(A)**, etc.
    # This handles cases like "**A**\n\n**Reasoning:**..."
    bold_match = re.search(r"^\s*\*\*\s*\(?\s*([A-Z])\s*\)?\s*\*\*", answer)
    if bold_match:
        return bold_match.group(1)

    # Check for quoted format: "C", "A", etc.
    # This handles cases like "\"B\"" or "\"A\"\n\nExplanation..."
    quote_match = re.search(r'^\s*["\']?\s*([A-Z])\s*["\']?(?:\s|$)', answer)
    if quote_match and (answer.strip().startswith('"') or answer.strip().startswith("'")):
        return quote_match.group(1)

    # Check for parentheses format: (C), (A), etc.
    # This handles cases like "(C)" or "(A)\n\nExplanation..."
    paren_match = re.search(r"^\s*\(\s*([A-Z])\s*\)", answer)
    if paren_match:
        return paren_match.group(1)

    # Check for letter with period at the start, possibly followed by reasoning
    # This handles cases like "B.\n\n**Reasoning:**..."
    letter_period_match = re.search(r"^\s*([A-Z])\.", answer)
    if letter_period_match:
        return letter_period_match.group(1)

    # Try to match markdown/formatted patterns: **(A)**, **B**, *(C)*, (D), etc.
    # This regex strips common markdown and finds the letter inside
    formatted_match = re.search(r"^\s*[*_]*\(?\s*([A-Z])\s*\)?[*_]*(?:\.|[^A-Z]|$)", answer)
    if formatted_match:
        return formatted_match.group(1)

    # Extract first capital letter followed by optional period, space, or end
    match = re.search(r"^\s*([A-Z])(?:\.|[\s,]|$)", answer)
    if match:
        return match.group(1)

    return ""


def extract_labels(answer: str) -> list[str]:
    """
    Extract multiple letter labels from a model's answer.

    This function handles cases where multiple correct answers are expected.
    It extracts all capital letters (A, B, C, D, etc.) found in the answer,
    including from answers with reasoning followed by "answer is" statements.

    Handles multiple formats:
    - Simple letters: "A", "B, C", "A,B,C"
    - Consecutive letters: "ABCD", "AB\n..."
    - Letters with periods: "A. B.", "D."
    - Letters with text: "A. text, B. more text"
    - With prefix: "Answer: A, C"
    - Bold comma-separated with prefix: "Answer: **A, C**"
    - Quoted format: "\"A\", \"C\"", "'B' and 'D'"
    - Parentheses format: "(A), (C)", "(B) and (D)"
    - Bold markdown: "**B**", "**A**, **C**", "**A. 4.65**", "**(A), (C)**"
    - Italic markdown: "*A*, *B*", "*(D)*"
    - With reasoning: "...reasoning... the answers are **(A)** and **(C)**."
    - Numbered list format with dash: "1. **Question?**\n   - A, C"

    Args:
        answer: The model's answer string

    Returns:
        List of extracted letter labels, preserving order and removing duplicates.
        Returns empty list if no valid labels are found.

    Examples:
        >>> extract_labels("A, B, C")
        ['A', 'B', 'C']
        >>> extract_labels("Answer: A. Text here, B. More text")
        ['A', 'B']
        >>> extract_labels("D")
        ['D']
        >>> extract_labels("**B**")
        ['B']
        >>> extract_labels("**A**, **C**")
        ['A', 'C']
        >>> extract_labels("(A), (C)")
        ['A', 'C']
        >>> extract_labels("\"A\", \"C\"")
        ['A', 'C']
        >>> extract_labels("...reasoning... the answers are **(A)** and **(C)**.")
        ['A', 'C']
        >>> extract_labels("1. **Question?**\\n   - A, C")
        ['A', 'C']
        >>> extract_labels("Answer: **A, C**")
        ['A', 'C']
        >>> extract_labels("B. No\\n\\n**Reasoning:**\\n\\nThe answer accurately describes...")
        ['B']
        >>> extract_labels("ABCD")
        ['A', 'B', 'C', 'D']
        >>> extract_labels("AB\\nInstead of 'satellites', the instruments deployed...")
        ['A', 'B']
        >>> extract_labels("...long CoT reasoning...\\nA, B")
        ['A', 'B']
    """
    # First, check for bare letter(s) on the very last line (explain-then-answer CoT pattern).
    # E.g., "...long reasoning...\nA, B" or "...long reasoning...\nA"
    # Letters may be separated by commas, "and", "&", or "/".
    # This must run first to avoid the rest of the logic being confused by uppercase letters
    # scattered throughout the reasoning text.
    bare_labels_end = re.search(
        r"(?:^|\n)\s*([A-Z](?:\s*(?:,|and|&|/)\s*[A-Z])*)\s*$", answer
    )
    if bare_labels_end:
        letters = re.findall(r"[A-Z]", bare_labels_end.group(1))
        seen = set()
        labels_list = []
        for letter in letters:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Check for comma-separated letters at the very START of the text (answer-first artifact).
    # Handles cases like " A, B, C, DOkay..." where letters are prepended before reasoning prose.
    # Requires at least two letters to avoid false positives on normal answer-first single letters.
    leading_csv_match = re.match(
        r"^\s*([A-Z](?:\s*,\s*[A-Z])+)(?:[A-Z][a-z]|\s)", answer
    )
    if leading_csv_match:
        letters = re.findall(r"[A-Z]", leading_csv_match.group(1))
        seen = set()
        labels_list = []
        for letter in letters:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Check for "Answer..." at the end of the text in various formats:
    # - Bold with colon:  "**Answer: A, B**"
    # - Plain with colon: "Answer: A, B"
    # - No colon/space:   "AnswerA, B"  (model artifact)
    # E.g., bullet-point analysis followed by one of these on the last line.
    bold_answer_end = re.search(
        r"(?:^|\n)\s*\*{0,2}\s*[Aa]nswer:?\s*\*{0,2}\s*([A-Z](?:\s*(?:,|and|&|/)\s*[A-Z])*)\s*\*{0,2}\s*$", answer
    )
    if not bold_answer_end:
        # Also match inline bold letters at end of last line after any prose:
        # E.g., "...the correct options are **A, B, C**."
        bold_answer_end = re.search(
            r"\*\*\s*([A-Z](?:\s*,\s*[A-Z])+)\s*\*\*[.\s]*$", answer
        )
    if bold_answer_end:
        letters = re.findall(r"[A-Z]", bold_answer_end.group(1))
        seen = set()
        labels_list = []
        for letter in letters:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Second, try to find numbered list format with dash-prefixed answers
    # Pattern: optional numbering/bullets followed by question, then dash with letters
    # E.g., "1. **Question?**\n   - A, C" or "2. **Question**\n   - B"
    # Use findall to get all matches, then take the last one (most likely the actual question)
    # Use negative lookahead to avoid matching "- Option A" (letter followed by lowercase)
    dash_answer_matches = re.findall(
        r"(?:^|\n)\s*\d*\.?\s*\*{0,2}[^*\n]*\*{0,2}\s*\n\s*-\s*([A-Z](?![a-z])(?:\s*,\s*[A-Z](?![a-z]))*)",
        answer,
        flags=re.MULTILINE
    )
    if dash_answer_matches:
        # Extract letters from the last dash-prefixed answer
        letters_text = dash_answer_matches[-1]
        labels_list = [letter.strip() for letter in letters_text.split(",")]
        return labels_list

    # Second, try to find "answer is/are" pattern and extract from that portion
    # Only apply this if the pattern appears before any bold section headers (Reasoning, Explanation, etc.)
    # to avoid matching descriptive text like "the answer is accurate" within explanations
    answer_before_headers = re.split(r'\*\*(?:Reasoning|Explanation|Rationale|Analysis):', answer)[0]
    answer_is_match = re.search(
        r"(?:the\s+)?answers?\s+(?:is|are)\s+(.+?)(?:\.|$)", answer_before_headers, flags=re.IGNORECASE | re.DOTALL
    )
    if answer_is_match:
        # Extract from the "answer is/are" portion only
        # But only if it contains letter labels (A-Z), not descriptive text
        extracted = answer_is_match.group(1)
        if re.search(r'[A-Z]', extracted):
            answer = extracted

    # Strip common prefixes at the beginning like "Answer: ", "answer: ", etc.
    answer = re.sub(
        r"^(?:Answer|answer|The answer is|the answer is):\s*", "", answer, flags=re.IGNORECASE
    )

    # Check for bold markdown with comma-separated letters at the start: **A, C**, **A, B, C**, etc.
    # This handles cases like "Answer: **A, C**" after the prefix has been stripped above.
    bold_csv_match = re.search(r"^\s*\*\*\s*([A-Z](?:\s*,\s*[A-Z])+)\s*\*\*", answer)
    if bold_csv_match:
        labels_list = [letter.strip() for letter in bold_csv_match.group(1).split(",")]
        return labels_list

    # First, check for consecutive capital letters at the start (e.g., "ABCD", "AB\n...")
    # This handles cases where multiple answers are given as a single string without separators
    consecutive_match = re.search(r"^\s*([A-Z]{2,})(?:\n|\.|\s|$)", answer)
    if consecutive_match:
        # Extract each letter from the consecutive string
        letters = list(consecutive_match.group(1))
        return letters

    # Second, check for simple letter pattern at the start of the answer
    # This pattern matches: "A", "A.", "B. No", etc. at the beginning
    # This should be checked before bold patterns to avoid matching section headers like "**Reasoning:**"
    start_letter_pattern = r"^\s*([A-Z])\.?"
    start_match = re.search(start_letter_pattern, answer)
    if start_match:
        # Found a letter at the start
        # Extract the first line or portion before bold section headers to avoid extracting letters from headers
        # Split on double newline or bold section headers (like **Reasoning:**, **Explanation:**, **Rationale:**)
        answer_before_header = re.split(r'\n\n|\*\*(?:Reasoning|Explanation|Rationale|Analysis):', answer)[0]

        # Now find all similar letters only in this portion
        letter_pattern = r"\b([A-Z])\.?(?:\s|,|and|&|$)"
        matches = re.findall(letter_pattern, answer_before_header)
        if matches:
            # Remove duplicates while preserving order
            seen = set()
            labels_list = []
            for letter in matches:
                if letter not in seen:
                    seen.add(letter)
                    labels_list.append(letter)
            return labels_list

    # Second, try to find quoted format: "A", "C", etc.
    # This handles cases like "\"A\", \"C\"" or "\"B\" and \"D\""
    quote_matches = re.findall(r'["\']([A-Z])["\']', answer)
    if quote_matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in quote_matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Third, try to find parentheses format: (A), (C), etc.
    # This handles cases like "(A), (C)" or "(B) and (D)"
    paren_matches = re.findall(r"\(\s*([A-Z])\s*\)", answer)
    if paren_matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in paren_matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Fourth, try to find bold markdown labels: **A**, **B**, **(A)**, **A. 4.65**, etc.
    # Pattern handles optional parentheses and other characters between ** markers
    # Use negative lookahead (?![a-z]) to ensure we don't match section headers like **Explanation:**
    bold_matches = re.findall(r"\*\*\s*\(?\s*([A-Z])(?![a-z])[^*]*\*\*", answer)
    if bold_matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in bold_matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Find all single capital letters followed by optional period
    # This pattern matches: "A", "A.", "D.", etc.
    letter_pattern = r"\b([A-Z])\.?(?:\s|,|and|&|$)"
    matches = re.findall(letter_pattern, answer)

    if matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Fallback: split by comma and strip
    labels_list = answer.split(",")
    labels_list = [label.strip() for label in labels_list]
    return labels_list


# Regex pattern for use in YAML configuration files
# This pattern extracts a single capital letter label
# Handles multiple formats including reasoning followed by "answer is" statements
# Formats: (A, A., Answer: A), bold markdown (**A**, **A. 4.65**, **(B)**),
# and "answer is" patterns (the answer is **(B)**.)
LABEL_EXTRACTION_REGEX = r"(?:(?:the\s+)?answer\s+is\s+[*_()]*([A-Z])[*_().\s]*|^\s*(?:Answer:\s*)?[*_]*\(?\s*([A-Z])\s*\)?[*_]*(?:\.|[^A-Z]|$))"

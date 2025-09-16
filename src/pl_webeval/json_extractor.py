"""
Robust JSON extraction utilities from LLM responses.
Based on the battle-tested logic from ModularV2.
"""

import json
import re
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def extract_first_json(text: str, context: str = "generic") -> Optional[Dict]:
    """
    Robustly extract the first JSON object from an LLM response text.
    Handles code fences and attempts balanced-brace extraction. Returns None on failure.
    
    Args:
        text: The text to extract JSON from
        context: Short label used in diagnostics/logs
    
    Returns:
        Extracted JSON as dict, or None if extraction fails
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided to extract_first_json")
        return None
    
    s = text.strip()
    
    # Strip markdown code fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.startswith("json"):
                s = s[4:].strip()
    
    # Try direct JSON parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")
    except Exception as e:
        logger.debug(f"Unexpected error in direct JSON parse: {e}")
    
    def fix_common_json_issues(json_str: str) -> str:
        # Fix unescaped newlines within strings
        json_str = re.sub(r'(?<!\\)\n', '\\n', json_str)
        # Add missing commas between properties
        json_str = re.sub(r'}\s*"{', '},"', json_str)
        return json_str
    
    # Try fixing and parsing
    fixed_s = fix_common_json_issues(s)
    try:
        result = json.loads(fixed_s)
        logger.info("Successfully parsed JSON after fixing newlines and structure")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed after fixing: {e}")
        # Try trimming to the last complete object end '}' in case of truncation
        try:
            last_brace = fixed_s.rfind('}')
            while last_brace != -1:
                candidate = fixed_s[:last_brace+1]
                try:
                    result = json.loads(candidate)
                    logger.info("Parsed JSON by trimming to last complete object boundary")
                    return result
                except Exception:
                    last_brace = fixed_s.rfind('}', 0, last_brace)
        except Exception as te:
            logger.debug(f"Trim parse attempt error: {te}")
    except Exception as e:
        logger.debug(f"Unexpected error after fixing JSON: {e}")

    # Progressive truncation fallback for trailing incomplete property
    try:
        truncated = fixed_s
        for _ in range(6):
            comma = truncated.rfind(',')
            if comma == -1:
                break
            truncated = truncated[:comma]
            truncated = truncated.rstrip()
            # Remove trailing colon if present (incomplete "key":)
            if truncated.endswith(':'):
                truncated = truncated[:-1]
            # Balance braces/brackets
            def _balance_end(t: str) -> str:
                brace_diff = t.count('{') - t.count('}')
                bracket_diff = t.count('[') - t.count(']')
                if brace_diff > 0:
                    t += '}' * brace_diff
                if bracket_diff > 0:
                    t += ']' * bracket_diff
                return t
            candidate2 = _balance_end(truncated)
            try:
                result = json.loads(candidate2)
                logger.info("Parsed JSON using progressive truncation fallback")
                return result
            except Exception:
                continue
    except Exception as te2:
        logger.debug(f"Progressive truncation attempt error: {te2}")

    # Balanced braces search for first JSON object with fix
    start = s.find("{")
    if start != -1:
        stack = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    stack += 1
                elif ch == "}":
                    stack -= 1
                    if stack == 0:
                        candidate = s[start:i+1]
                        try:
                            # Try with fixing first
                            fixed_candidate = fix_common_json_issues(candidate)
                            return json.loads(fixed_candidate)
                        except Exception:
                            try:
                                # Try without fixing
                                return json.loads(candidate)
                            except Exception:
                                break  # stop searching further

    # Enhanced comma-fixing function for missing delimiters
    def fix_missing_commas(json_str: str) -> str:
        """Fix missing commas between JSON properties by analyzing line-by-line structure."""
        lines = json_str.split('\n')
        fixed_lines = []
        in_string = False
        escape_next = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                fixed_lines.append(line)
                continue

            # Track string state
            temp_line = []
            for char in line:
                if escape_next:
                    temp_line.append(char)
                    escape_next = False
                    continue
                if char == '\\':
                    temp_line.append(char)
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                temp_line.append(char)

            fixed_lines.append(''.join(temp_line))

            # Check if we need to add comma after this line
            if i < len(lines) - 1 and not in_string:
                next_line = lines[i + 1].strip()
                # If current line ends with a value and next line starts with a key
                if (stripped.endswith(('true', 'false', 'null', ']', '}', '"')) and
                   next_line.startswith('"') and not next_line.endswith((',', '{', '['))):
                    # Look ahead to see if this is a property continuation
                    if not (stripped.endswith('}') and next_line.startswith('}')) and \
                       not (stripped.endswith(']') and next_line.startswith(']')):
                        fixed_lines[-1] += ','

        return '\n'.join(fixed_lines)

    # Last-chance fixes for common non-JSON literals
    try:
        s2 = re.sub(r'\bTrue\b', 'true', s)
        s2 = re.sub(r'\bFalse\b', 'false', s2)
        s2 = re.sub(r'\bNone\b', 'null', s2)
        # Also try to fix newlines in this version
        s2 = fix_common_json_issues(s2)
        result = json.loads(s2)
        logger.info("Successfully parsed JSON with last-chance fixes")
        return result
    except Exception as e:
        # Before giving up, attempt a structural repair that:
        # - Closes an unterminated string
        # - Removes trailing commas before closing braces/brackets
        # - Balances unmatched {} and []
        def _repair_and_balance(doc: str) -> str:
            try:
                in_string = False
                esc = False
                brace = 0
                bracket = 0
                out_chars = []
                for ch in doc:
                    out_chars.append(ch)
                    if in_string:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == '{':
                            brace += 1
                        elif ch == '}':
                            brace = max(0, brace - 1)
                        elif ch == '[':
                            bracket += 1
                        elif ch == ']':
                            bracket = max(0, bracket - 1)
                repaired = ''.join(out_chars)
                # Remove trailing commas before closers (e.g., "key": "val", } -> "key":"val" })
                repaired = re.sub(r',\s*(\}|\])', r'\1', repaired)
                # Close any unterminated string at EOF
                if in_string:
                    repaired += '"'
                # Balance braces/brackets
                if brace > 0:
                    repaired += '}' * brace
                if bracket > 0:
                    repaired += ']' * bracket
                # Final cleanup: if we created },] mismatches due to order, try minimal normalization
                repaired = re.sub(r',\s*(\}|\])', r'\1', repaired)
                return repaired
            except Exception:
                return doc

        # Choose the best candidate document to repair
        candidate_doc = s2 if 's2' in locals() else (fixed_s if 'fixed_s' in locals() else s)
        repaired_doc = _repair_and_balance(candidate_doc)
        try:
            return json.loads(repaired_doc)
        except Exception as e2:
            logger.error(f"Failed to parse JSON with all attempts: {e2}")
            # Print precise failure location and a context window with a caret (^) at the failure position
            try:
                if isinstance(e2, json.JSONDecodeError):
                    pos = getattr(e2, 'pos', None)
                    lineno = getattr(e2, 'lineno', None)
                    colno = getattr(e2, 'colno', None)
                    doc = repaired_doc
                    total_len = len(doc)
                    if isinstance(pos, int):
                        start = max(0, pos - 120)
                        end = min(total_len, pos + 120)
                        snippet = doc[start:end]
                        caret = ' ' * (pos - start) + '^'
                        print(f"---- JSON parse failure context ({context}) ----")
                        print(f"len={total_len} pos={pos} line={lineno} col={colno}")
                        print(snippet)
                        print(caret)
                        print("--------------------------------------------------------")
                    else:
                        print("JSON parse error without position; preview of document:")
                        print(doc[:500])
                else:
                    # Non-JSONDecodeError: provide a preview of the original text
                    print("Non-JSONDecodeError; preview of original text:")
                    print((text or "")[:500])
            except Exception as _ctx_e:
                logger.debug(f"Failed to render JSON failure context: {_ctx_e}")
            return None
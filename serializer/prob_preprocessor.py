"""
Preprocessor for probabilistic HDDL extensions.

Reads an HDDL domain file, finds `probabilistic` blocks, extracts probability
maps, and rewrites them as standard `oneof` blocks for downstream processing.
"""

import os
import tempfile


def _find_matching_paren(text, start):
    """Find the index of the closing paren matching the open paren at `start`."""
    assert text[start] == '('
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched parenthesis at position {start}")


def _parse_probabilistic_block(block_text):
    """
    Parse a probabilistic block and return (probabilities, cleaned_oneof_text).

    Input:  '(probabilistic 0.8 (on ?l) 0.2 (off ?l))'
    Output: ([0.8, 0.2], '(oneof (on ?l) (off ?l))')
    """
    inner = block_text.strip()
    assert inner.startswith('(probabilistic')
    # Strip '(probabilistic' prefix and trailing ')'
    inner = inner[len('(probabilistic'):-1].strip()

    probabilities = []
    effects = []
    i = 0
    while i < len(inner):
        # Skip whitespace
        while i < len(inner) and inner[i].isspace():
            i += 1
        if i >= len(inner):
            break
        # Read probability number (stops at whitespace or '(')
        j = i
        while j < len(inner) and not inner[j].isspace() and inner[j] != '(':
            j += 1
        prob = float(inner[i:j])
        probabilities.append(prob)
        i = j
        # Skip whitespace
        while i < len(inner) and inner[i].isspace():
            i += 1
        # Read effect: balanced parens starting at '('
        assert inner[i] == '(', f"Expected '(' for effect, got '{inner[i]}' at position {i}"
        end = _find_matching_paren(inner, i)
        effects.append(inner[i:end + 1])
        i = end + 1

    cleaned = '(oneof ' + ' '.join(effects) + ')'
    return probabilities, cleaned


def _find_action_name(text, probabilistic_pos):
    """
    Given the position of a probabilistic block, find the enclosing (:action ...) name.
    Searches backwards for (:action <name>).
    """
    search_region = text[:probabilistic_pos]
    # Find the last occurrence of ':action'
    idx = search_region.rfind(':action')
    if idx == -1:
        return None
    # Extract the action name (next token after ':action')
    rest = search_region[idx + len(':action'):].strip()
    name = rest.split()[0] if rest.split() else None
    return name


def preprocess(domain_path):
    """
    Preprocess an HDDL domain file containing `probabilistic` blocks.

    Args:
        domain_path: Path to the HDDL domain file.

    Returns:
        (temp_file_path, prob_map) where:
        - temp_file_path: Path to cleaned HDDL file with probabilistic replaced by oneof
        - prob_map: dict mapping ungrounded action names to probability lists
                    e.g. {"observe": [0.8, 0.2]}
    """
    with open(domain_path, 'r') as f:
        text = f.read()

    prob_map = {}
    cleaned_text = text

    # Find all probabilistic occurrences (working from end to start to preserve positions)
    positions = []
    search_start = 0
    while True:
        idx = cleaned_text.find('probabilistic', search_start)
        if idx == -1:
            break
        # Find the opening paren before 'probabilistic'
        open_paren = idx - 1
        while open_paren >= 0 and cleaned_text[open_paren] in ' \t\n':
            open_paren -= 1
        if open_paren >= 0 and cleaned_text[open_paren] == '(':
            positions.append(open_paren)
        search_start = idx + len('probabilistic')

    # Process from end to start to preserve earlier positions
    for open_paren in reversed(positions):
        close_paren = _find_matching_paren(cleaned_text, open_paren)
        block = cleaned_text[open_paren:close_paren + 1]
        action_name = _find_action_name(cleaned_text, open_paren)
        probabilities, oneof_text = _parse_probabilistic_block(block)
        if action_name:
            prob_map[action_name] = probabilities
        cleaned_text = cleaned_text[:open_paren] + oneof_text + cleaned_text[close_paren + 1:]

    # Write cleaned HDDL to a temp file
    fd, temp_path = tempfile.mkstemp(suffix='.hddl', prefix='prob_cleaned_')
    with os.fdopen(fd, 'w') as f:
        f.write(cleaned_text)

    return temp_path, prob_map


def has_probabilistic(domain_path):
    """Check if a domain file contains probabilistic blocks."""
    with open(domain_path, 'r') as f:
        return 'probabilistic' in f.read()

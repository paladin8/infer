"""Regex to finite-state machine compiler.

Compiles regex patterns into NFAs (Thompson construction), then converts
to DFAs (subset construction) for efficient token-level guidance.

Supported regex features:
- Literal characters (with backslash escaping)
- Character classes: [abc], [a-z], [^abc], \\d, \\w, \\s and negations
- Quantifiers: *, +, ?, {n}, {n,m}, {n,}
- Alternation: a|b
- Grouping: (...)
- Dot: . (matches any char except newline)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# NFA representation
# ---------------------------------------------------------------------------


@dataclass
class NFAState:
    """A state in a Thompson NFA.

    Attributes:
        transitions: Mapping from character to list of target state IDs.
        epsilon: List of state IDs reachable via epsilon transitions.
        is_accept: Whether this is an accepting state.
    """

    transitions: dict[str, list[int]] = field(default_factory=dict)
    epsilon: list[int] = field(default_factory=list)
    is_accept: bool = False


@dataclass
class NFA:
    """Thompson NFA with a single start and single accept state.

    Attributes:
        states: All NFA states, indexed by position.
        start: Index of the start state.
        accept: Index of the accept state.
    """

    states: list[NFAState]
    start: int
    accept: int


# ---------------------------------------------------------------------------
# DFA representation
# ---------------------------------------------------------------------------


@dataclass
class DFA:
    """Deterministic finite automaton.

    Attributes:
        transitions: state -> char -> next_state mapping.
        initial_state: Starting state ID.
        accept_states: Set of accepting state IDs.
        alphabet: Set of all characters in the DFA's alphabet.
    """

    transitions: dict[int, dict[str, int]]
    initial_state: int
    accept_states: set[int]
    alphabet: set[str]

    def accepts(self, text: str) -> bool:
        """Check whether the DFA accepts a string.

        Args:
            text: Input string to test.

        Returns:
            True if the string is accepted.
        """
        state = self.initial_state
        for ch in text:
            trans = self.transitions.get(state)
            if trans is None or ch not in trans:
                return False
            state = trans[ch]
        return state in self.accept_states

    def walk(self, text: str, start_state: int | None = None) -> int | None:
        """Walk the DFA on a string, returning the final state or None if stuck.

        Args:
            text: Input string to walk.
            start_state: Starting state (defaults to initial_state).

        Returns:
            Final state ID, or None if a dead transition was encountered.
        """
        state = start_state if start_state is not None else self.initial_state
        for ch in text:
            trans = self.transitions.get(state)
            if trans is None or ch not in trans:
                return None
            state = trans[ch]
        return state


# ---------------------------------------------------------------------------
# NFA construction helpers (Thompson construction)
# ---------------------------------------------------------------------------


def _new_nfa(start_state: NFAState, accept_state: NFAState) -> NFA:
    """Create a minimal 2-state NFA."""
    accept_state.is_accept = True
    return NFA(states=[start_state, accept_state], start=0, accept=1)


def _nfa_for_char(ch: str) -> NFA:
    """NFA that matches a single character."""
    start = NFAState()
    accept = NFAState(is_accept=True)
    start.transitions[ch] = [1]
    return NFA(states=[start, accept], start=0, accept=1)


def _nfa_for_chars(chars: set[str]) -> NFA:
    """NFA that matches any one of the given characters."""
    start = NFAState()
    accept = NFAState(is_accept=True)
    for ch in chars:
        if ch in start.transitions:
            start.transitions[ch].append(1)
        else:
            start.transitions[ch] = [1]
    return NFA(states=[start, accept], start=0, accept=1)


def _nfa_epsilon() -> NFA:
    """NFA that matches the empty string."""
    start = NFAState()
    accept = NFAState(is_accept=True)
    start.epsilon.append(1)
    return NFA(states=[start, accept], start=0, accept=1)


def _renumber_nfa(nfa: NFA, offset: int) -> None:
    """Shift all state references in an NFA by offset (in-place)."""
    for state in nfa.states:
        for ch in state.transitions:
            state.transitions[ch] = [t + offset for t in state.transitions[ch]]
        state.epsilon = [e + offset for e in state.epsilon]
    nfa.start += offset
    nfa.accept += offset


def _concat_nfa(a: NFA, b: NFA) -> NFA:
    """Concatenate two NFAs: L(a) followed by L(b)."""
    offset = len(a.states)
    _renumber_nfa(b, offset)

    # Merge accept of a into start of b via epsilon.
    a.states[a.accept].is_accept = False
    a.states[a.accept].epsilon.append(b.start)

    states = a.states + b.states
    return NFA(states=states, start=a.start, accept=b.accept)


def _alternate_nfa(a: NFA, b: NFA) -> NFA:
    """Alternation: L(a) | L(b)."""
    # Add new start and accept states.
    new_start = NFAState()
    new_accept = NFAState(is_accept=True)

    # Renumber a and b.
    _renumber_nfa(a, 1)  # offset by 1 for new_start
    b_offset = 1 + len(a.states)
    _renumber_nfa(b, b_offset)

    # Wire up epsilon transitions.
    new_start.epsilon = [a.start, b.start]
    a.states[a.accept - 1].is_accept = False  # clear old accept
    b.states[b.accept - b_offset].is_accept = False

    accept_idx = 1 + len(a.states) + len(b.states)
    a.states[a.accept - 1].epsilon.append(accept_idx)
    b.states[b.accept - b_offset].epsilon.append(accept_idx)

    states = [new_start, *a.states, *b.states, new_accept]
    return NFA(states=states, start=0, accept=accept_idx)


def _kleene_star_nfa(inner: NFA) -> NFA:
    """Kleene star: L(inner)*."""
    new_start = NFAState()
    new_accept = NFAState(is_accept=True)

    _renumber_nfa(inner, 1)

    accept_idx = 1 + len(inner.states)
    new_start.epsilon = [inner.start, accept_idx]

    inner.states[inner.accept - 1].is_accept = False
    inner.states[inner.accept - 1].epsilon.extend([inner.start, accept_idx])

    states = [new_start, *inner.states, new_accept]
    return NFA(states=states, start=0, accept=accept_idx)


def _kleene_plus_nfa(inner: NFA) -> NFA:
    """Kleene plus: L(inner)+. At least one occurrence."""
    new_start = NFAState()
    new_accept = NFAState(is_accept=True)

    _renumber_nfa(inner, 1)

    accept_idx = 1 + len(inner.states)
    new_start.epsilon = [inner.start]

    inner.states[inner.accept - 1].is_accept = False
    inner.states[inner.accept - 1].epsilon.extend([inner.start, accept_idx])

    states = [new_start, *inner.states, new_accept]
    return NFA(states=states, start=0, accept=accept_idx)


def _optional_nfa(inner: NFA) -> NFA:
    """Optional: L(inner)?. Zero or one occurrence."""
    new_start = NFAState()
    new_accept = NFAState(is_accept=True)

    _renumber_nfa(inner, 1)

    accept_idx = 1 + len(inner.states)
    new_start.epsilon = [inner.start, accept_idx]

    inner.states[inner.accept - 1].is_accept = False
    inner.states[inner.accept - 1].epsilon.append(accept_idx)

    states = [new_start, *inner.states, new_accept]
    return NFA(states=states, start=0, accept=accept_idx)


def _repeat_nfa(inner_factory: object, min_count: int, max_count: int | None) -> NFA:
    """Repeat: {min,max}. If max is None, it's {min,}."""
    # Build the required part: inner{min}
    if min_count == 0:
        result = _nfa_epsilon()
    else:
        # Concatenate min copies.
        parts: list[NFA] = []
        for _ in range(min_count):
            parts.append(inner_factory())  # type: ignore[operator]
        result = parts[0]
        for p in parts[1:]:
            result = _concat_nfa(result, p)

    if max_count is None:
        # {min,}: required part + star
        star = _kleene_star_nfa(inner_factory())  # type: ignore[operator]
        result = _concat_nfa(result, star)
    elif max_count > min_count:
        # {min,max}: required part + optional copies
        for _ in range(max_count - min_count):
            opt = _optional_nfa(inner_factory())  # type: ignore[operator]
            result = _concat_nfa(result, opt)

    return result


# ---------------------------------------------------------------------------
# Regex parser
# ---------------------------------------------------------------------------

# Printable ASCII characters (space through tilde).
_PRINTABLE_ASCII = set(chr(i) for i in range(32, 127))

# Common whitespace characters.
_WHITESPACE = {" ", "\t", "\n", "\r"}

# Shorthand character classes.
_DIGIT_CHARS = set("0123456789")
_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE_CHARS = _WHITESPACE

# Characters that need escaping in regex context.
_SPECIAL_CHARS = set("\\[](){}|*+?.")


class _RegexParser:
    """Recursive descent parser for regex patterns.

    Parses a regex into an NFA using Thompson construction.
    Grammar:
        expr     -> term ('|' term)*
        term     -> factor+
        factor   -> atom quantifier?
        atom     -> '(' expr ')' | char_class | '.' | literal
        quantifier -> '*' | '+' | '?' | '{' n (',' m?)? '}'
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0

    def peek(self) -> str | None:
        """Look at the next character without consuming it."""
        if self.pos >= len(self.pattern):
            return None
        return self.pattern[self.pos]

    def advance(self) -> str:
        """Consume and return the next character."""
        if self.pos >= len(self.pattern):
            raise ValueError(
                f"Unexpected end of pattern at position {self.pos} in {self.pattern!r}"
            )
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def expect(self, ch: str) -> None:
        """Consume the next character, asserting it matches."""
        if self.pos >= len(self.pattern):
            raise ValueError(
                f"Expected {ch!r} at position {self.pos}, but pattern ended in {self.pattern!r}"
            )
        actual = self.advance()
        if actual != ch:
            raise ValueError(
                f"Expected {ch!r} at position {self.pos - 1}, got {actual!r} in {self.pattern!r}"
            )

    def parse(self) -> NFA:
        """Parse the full pattern into an NFA."""
        if len(self.pattern) == 0:
            return _nfa_epsilon()
        nfa = self._parse_expr()
        if self.pos < len(self.pattern):
            raise ValueError(
                f"Unexpected character {self.pattern[self.pos]!r} "
                f"at position {self.pos} in {self.pattern!r}"
            )
        return nfa

    def _parse_expr(self) -> NFA:
        """Parse alternation: term ('|' term)*."""
        left = self._parse_term()
        while self.peek() == "|":
            self.advance()
            right = self._parse_term()
            left = _alternate_nfa(left, right)
        return left

    def _parse_term(self) -> NFA:
        """Parse concatenation: factor+."""
        factors: list[NFA] = []
        while self.peek() is not None and self.peek() not in ("|", ")"):
            factors.append(self._parse_factor())
        if not factors:
            return _nfa_epsilon()
        result = factors[0]
        for f in factors[1:]:
            result = _concat_nfa(result, f)
        return result

    def _parse_factor(self) -> NFA:
        """Parse factor: atom quantifier?."""
        # Save position to create a factory for repeat quantifiers.
        atom_start = self.pos
        atom = self._parse_atom()
        atom_end = self.pos

        quant = self._parse_quantifier()
        if quant is None:
            return atom

        kind, min_c, max_c = quant
        if kind == "*":
            return _kleene_star_nfa(atom)
        elif kind == "+":
            return _kleene_plus_nfa(atom)
        elif kind == "?":
            return _optional_nfa(atom)
        elif kind == "repeat":
            # Need a factory that re-parses the atom.
            atom_pattern = self.pattern[atom_start:atom_end]

            def make_atom(p: str = atom_pattern) -> NFA:
                parser = _RegexParser(p)
                return parser._parse_atom()

            return _repeat_nfa(make_atom, min_c, max_c)
        return atom  # pragma: no cover

    def _parse_quantifier(self) -> tuple[str, int, int | None] | None:
        """Parse quantifier: *, +, ?, {n}, {n,}, {n,m}."""
        ch = self.peek()
        if ch == "*":
            self.advance()
            return ("*", 0, None)
        elif ch == "+":
            self.advance()
            return ("+", 1, None)
        elif ch == "?":
            self.advance()
            return ("?", 0, 1)
        elif ch == "{":
            self.advance()
            n = self._parse_int()
            if self.peek() == ",":
                self.advance()
                if self.peek() == "}":
                    self.expect("}")
                    return ("repeat", n, None)
                m = self._parse_int()
                self.expect("}")
                return ("repeat", n, m)
            self.expect("}")
            return ("repeat", n, n)
        return None

    def _parse_int(self) -> int:
        """Parse an integer from the current position."""
        digits = ""
        while self.peek() is not None and self.peek().isdigit():  # type: ignore[union-attr]
            digits += self.advance()
        if not digits:
            raise ValueError(f"Expected integer at position {self.pos} in {self.pattern!r}")
        return int(digits)

    def _parse_atom(self) -> NFA:
        """Parse atom: group, char class, dot, or literal."""
        ch = self.peek()
        if ch == "(":
            self.advance()
            nfa = self._parse_expr()
            self.expect(")")
            return nfa
        elif ch == "[":
            return self._parse_char_class()
        elif ch == ".":
            self.advance()
            # Match any printable ASCII + common whitespace except newline.
            dot_chars = (_PRINTABLE_ASCII | _WHITESPACE) - {"\n"}
            return _nfa_for_chars(dot_chars)
        elif ch == "\\":
            return self._parse_escape()
        elif ch is not None and ch not in _SPECIAL_CHARS:
            self.advance()
            return _nfa_for_char(ch)
        elif ch is not None and ch in (")", "|"):
            # These are handled by callers.
            raise ValueError(f"Unexpected {ch!r} at position {self.pos} in {self.pattern!r}")
        else:
            raise ValueError(
                f"Unexpected character {ch!r} at position {self.pos} in {self.pattern!r}"
            )

    def _parse_escape(self) -> NFA:
        """Parse an escaped character or shorthand class."""
        self.expect("\\")
        ch = self.advance()
        if ch == "d":
            return _nfa_for_chars(_DIGIT_CHARS)
        elif ch == "D":
            return _nfa_for_chars(_PRINTABLE_ASCII - _DIGIT_CHARS)
        elif ch == "w":
            return _nfa_for_chars(_WORD_CHARS)
        elif ch == "W":
            return _nfa_for_chars(_PRINTABLE_ASCII - _WORD_CHARS)
        elif ch == "s":
            return _nfa_for_chars(_SPACE_CHARS)
        elif ch == "S":
            return _nfa_for_chars((_PRINTABLE_ASCII | _WHITESPACE) - _SPACE_CHARS)
        elif ch == "n":
            return _nfa_for_char("\n")
        elif ch == "t":
            return _nfa_for_char("\t")
        elif ch == "r":
            return _nfa_for_char("\r")
        else:
            # Escaped literal (handles \\, \., \[, etc.)
            return _nfa_for_char(ch)

    def _parse_char_class(self) -> NFA:
        """Parse a character class: [abc], [a-z], [^abc]."""
        self.expect("[")
        negated = False
        if self.peek() == "^":
            self.advance()
            negated = True

        chars: set[str] = set()
        while self.peek() != "]" and self.peek() is not None:
            ch = self.advance()
            if ch == "\\":
                # Escaped character inside class.
                esc = self.advance()
                if esc == "d":
                    chars |= _DIGIT_CHARS
                elif esc == "D":
                    chars |= _PRINTABLE_ASCII - _DIGIT_CHARS
                elif esc == "w":
                    chars |= _WORD_CHARS
                elif esc == "W":
                    chars |= _PRINTABLE_ASCII - _WORD_CHARS
                elif esc == "s":
                    chars |= _SPACE_CHARS
                elif esc == "S":
                    chars |= (_PRINTABLE_ASCII | _WHITESPACE) - _SPACE_CHARS
                elif esc == "n":
                    chars.add("\n")
                elif esc == "t":
                    chars.add("\t")
                elif esc == "r":
                    chars.add("\r")
                else:
                    chars.add(esc)
            elif (
                self.peek() == "-"
                and self.pos + 1 < len(self.pattern)
                and self.pattern[self.pos + 1] != "]"
            ):
                # Range: a-z
                self.advance()  # consume '-'
                end_ch = self.advance()
                for i in range(ord(ch), ord(end_ch) + 1):
                    chars.add(chr(i))
            else:
                chars.add(ch)

        self.expect("]")

        if negated:
            chars = (_PRINTABLE_ASCII | _WHITESPACE) - chars

        return _nfa_for_chars(chars)


# ---------------------------------------------------------------------------
# NFA to DFA conversion (subset construction)
# ---------------------------------------------------------------------------


def _epsilon_closure(nfa: NFA, state_ids: frozenset[int]) -> frozenset[int]:
    """Compute epsilon closure of a set of NFA states."""
    stack = list(state_ids)
    closure = set(state_ids)
    while stack:
        s = stack.pop()
        for eps_target in nfa.states[s].epsilon:
            if eps_target not in closure:
                closure.add(eps_target)
                stack.append(eps_target)
    return frozenset(closure)


def _move(nfa: NFA, state_ids: frozenset[int], ch: str) -> frozenset[int]:
    """Compute the set of states reachable from state_ids on character ch."""
    result: set[int] = set()
    for s in state_ids:
        targets = nfa.states[s].transitions.get(ch, [])
        result.update(targets)
    return frozenset(result)


def nfa_to_dfa(nfa: NFA) -> DFA:
    """Convert an NFA to a DFA via subset construction.

    Args:
        nfa: The NFA to convert.

    Returns:
        An equivalent DFA.
    """
    # Collect alphabet from NFA.
    alphabet: set[str] = set()
    for state in nfa.states:
        alphabet.update(state.transitions.keys())

    initial_closure = _epsilon_closure(nfa, frozenset([nfa.start]))

    # Map from frozenset of NFA states -> DFA state ID.
    dfa_state_map: dict[frozenset[int], int] = {initial_closure: 0}
    dfa_transitions: dict[int, dict[str, int]] = {}
    dfa_accept: set[int] = set()
    worklist: list[frozenset[int]] = [initial_closure]
    next_id = 1

    if nfa.accept in initial_closure:
        dfa_accept.add(0)

    while worklist:
        current = worklist.pop()
        current_id = dfa_state_map[current]
        dfa_transitions[current_id] = {}

        for ch in sorted(alphabet):
            moved = _move(nfa, current, ch)
            if not moved:
                continue
            closure = _epsilon_closure(nfa, moved)
            if not closure:
                continue

            if closure not in dfa_state_map:
                dfa_state_map[closure] = next_id
                if nfa.accept in closure:
                    dfa_accept.add(next_id)
                worklist.append(closure)
                next_id += 1

            dfa_transitions[current_id][ch] = dfa_state_map[closure]

    return DFA(
        transitions=dfa_transitions,
        initial_state=0,
        accept_states=dfa_accept,
        alphabet=alphabet,
    )


# ---------------------------------------------------------------------------
# DFA minimization (remove unreachable and dead states)
# ---------------------------------------------------------------------------


def minimize_dfa(dfa: DFA) -> DFA:
    """Remove unreachable and dead states from a DFA.

    A state is unreachable if it cannot be reached from the initial state.
    A state is dead if no accepting state is reachable from it.

    Args:
        dfa: The DFA to minimize.

    Returns:
        A new DFA with unreachable and dead states removed.
    """
    # Find reachable states via BFS from initial.
    reachable: set[int] = set()
    queue = [dfa.initial_state]
    reachable.add(dfa.initial_state)
    while queue:
        s = queue.pop(0)
        for ch in dfa.transitions.get(s, {}):
            target = dfa.transitions[s][ch]
            if target not in reachable:
                reachable.add(target)
                queue.append(target)

    # Find live states (can reach an accepting state) via reverse BFS.
    # Build reverse graph.
    reverse: dict[int, set[int]] = {s: set() for s in reachable}
    for s in reachable:
        for _ch, t in dfa.transitions.get(s, {}).items():
            if t in reachable:
                reverse.setdefault(t, set()).add(s)

    live: set[int] = set()
    queue_live = [s for s in dfa.accept_states if s in reachable]
    live.update(queue_live)
    while queue_live:
        s = queue_live.pop(0)
        for pred in reverse.get(s, set()):
            if pred not in live:
                live.add(pred)
                queue_live.append(pred)

    # Keep states that are both reachable and live.
    keep = reachable & live

    # If initial state is not kept, the DFA accepts nothing.
    if dfa.initial_state not in keep:
        return DFA(
            transitions={0: {}},
            initial_state=0,
            accept_states=set(),
            alphabet=dfa.alphabet,
        )

    # Renumber states.
    old_to_new: dict[int, int] = {}
    new_id = 0
    # Ensure initial state is 0.
    old_to_new[dfa.initial_state] = 0
    new_id = 1
    for s in sorted(keep):
        if s != dfa.initial_state:
            old_to_new[s] = new_id
            new_id += 1

    new_transitions: dict[int, dict[str, int]] = {}
    for s in keep:
        new_s = old_to_new[s]
        new_transitions[new_s] = {}
        for ch, t in dfa.transitions.get(s, {}).items():
            if t in keep:
                new_transitions[new_s][ch] = old_to_new[t]

    new_accept = {old_to_new[s] for s in dfa.accept_states if s in keep}

    return DFA(
        transitions=new_transitions,
        initial_state=0,
        accept_states=new_accept,
        alphabet=dfa.alphabet,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_regex(pattern: str) -> NFA:
    """Parse a regex pattern into a Thompson NFA.

    Args:
        pattern: Regex pattern string.

    Returns:
        An NFA accepting the language defined by the pattern.

    Raises:
        ValueError: If the pattern is invalid.
    """
    parser = _RegexParser(pattern)
    return parser.parse()


def compile_regex(pattern: str) -> DFA:
    """Compile a regex pattern into a minimized DFA.

    Args:
        pattern: Regex pattern string.

    Returns:
        A minimized DFA accepting the language defined by the pattern.

    Raises:
        ValueError: If the pattern is invalid.
    """
    nfa = parse_regex(pattern)
    dfa = nfa_to_dfa(nfa)
    return minimize_dfa(dfa)

"""
SpacyParser - A reusable class for parsing natural language home automation commands.

Usage:
    from spacy_parser import SpacyParser

    parser = SpacyParser()
    results = parser.parse_as_tuples("turn on the lights please")
    # → [("light", "turn on")]
"""

import re
import json
import spacy
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedCommand:
    device: str
    action: str
    negated: bool = False
    source_text: str = field(default="", repr=False)

    def as_tuple(self) -> Tuple[str, str]:
        return (self.device, self.action)


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class SpacyParser:
    """
    Parses natural-language home-automation transcripts into (device, action) pairs.

    Parameters
    ----------
    config_path : str | Path | None
        Path to config.json. Defaults to config.json in the same directory as
        this file. Must contain "devices" and "actions" lists.
    model : str
        spaCy model name. Defaults to "en_core_web_sm".

    Example
    -------
    >>> parser = SpacyParser()
    >>> parser.parse_as_tuples("turn on the fan and switch off the heater")
    [('fan', 'turn on'), ('heater', 'switch off')]
    >>> parser.parse_as_tuples("turn on the lights please and also turn off the fan thanks!!")
    [('light', 'turn on'), ('fan', 'turn off')]
    """

    _PARTICLES = {"on", "off"}

    def __init__(
        self,
        config_path=None,
        model: str = "en_core_web_sm",
    ):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self._config_path = Path(config_path)
        self._model_name = model
        self._nlp = None
        self._devices: List[str] = []
        self._devices_lower: set = set()
        self._actions: List[str] = []

        self._load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, transcript: str) -> List[ParsedCommand]:
        """
        Parse a transcript and return a list of ParsedCommand objects.
        Filler words ("please", "thanks", etc.) are ignored automatically.
        """
        if not transcript or not transcript.strip():
            return []

        nlp = self._get_nlp()
        cleaned = self._preprocess(transcript)
        doc = nlp(cleaned)
        commands = self._extract_commands(doc)

        seen = set()
        unique = []
        for cmd in commands:
            key = (cmd.device, cmd.action)
            if key not in seen:
                seen.add(key)
                unique.append(cmd)

        return unique

    def parse_as_tuples(self, transcript: str) -> List[Tuple[str, str]]:
        """Convenience wrapper: returns list of (device, action) tuples."""
        return [cmd.as_tuple() for cmd in self.parse(transcript)]

    @property
    def devices(self) -> List[str]:
        return list(self._devices)

    @property
    def actions(self) -> List[str]:
        return list(self._actions)

    # ------------------------------------------------------------------
    # Config & model setup
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._config_path}\n"
                "Provide a config_path argument or place config.json next to this file."
            )

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for key in ("devices", "actions"):
            if key not in config:
                raise KeyError(f"config.json is missing required key: '{key}'")

        self._devices = [str(d).lower() for d in config["devices"]]
        self._devices_lower = set(self._devices)
        self._actions = [str(a).lower() for a in config["actions"]]

    def _get_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError:
                raise OSError(
                    f"spaCy model '{self._model_name}' not found.\n"
                    f"Install it with: python -m spacy download {self._model_name}"
                )
            self._add_device_ruler(self._nlp)
        return self._nlp

    def _add_device_ruler(self, nlp) -> None:
        patterns = [{"label": "DEVICE", "pattern": d} for d in self._devices]
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """
        Clean transcript text before sending to spaCy.
        - Strips/collapses whitespace
        - Inserts sentence breaks before chained action verbs
          so "turn on the light turn off the fan" parses as two sentences
        - Does NOT strip filler words — spaCy ignores them naturally
        """
        text = text.strip()
        text = " ".join(text.split())

        action_verbs = r"(?:turn|switch|activate|deactivate|enable|disable|start|stop)"

        # Insert period before a new verb phrase that follows a word character.
        # Using \w (not [^']) so contractions like "don't turn" aren't broken.
        text = re.sub(
            rf"(\w)\s+({action_verbs}\s+(?:on|off)?\s)",
            r"\1. \2",
            text,
            flags=re.IGNORECASE,
        )

        return text

    # ------------------------------------------------------------------
    # Dependency-tree extraction
    # ------------------------------------------------------------------

    def _extract_commands(self, doc) -> List[ParsedCommand]:
        """
        Walk the dependency tree and extract ParsedCommand objects.

        Handles:
        - Coordinated verbs:    "turn on the light and switch off the fan"
        - Coordinated objects:  "turn on the light and fan"
        - Plural forms:         "turn on the lights" → device "light"
        - Phrasal verbs:        "turn on", "switch off"
        - Negation:             "don't turn on the light"
        - VP ellipsis:          "turn on the light and off the fan"
        - Filler words:         "please", "thanks", "also" ignored naturally
        - Garbled input:        keyword fallback
        """
        commands: List[ParsedCommand] = []

        action_verbs = [
            tok for tok in doc
            if tok.pos_ == "VERB"
            and tok.dep_ in ("ROOT", "conj", "advcl", "xcomp", "ccomp")
        ]

        for verb in action_verbs:
            particle = self._get_particle(verb)
            negated = self._is_negated(verb)
            action_phrase = f"{verb.lemma_.lower()} {particle}".strip()

            for dev_tok in self._find_device_objects(verb):
                canonical = self._match_device(dev_tok)
                if canonical:
                    commands.append(ParsedCommand(
                        device=canonical,
                        action=action_phrase,
                        negated=negated,
                        source_text=doc.text,
                    ))

        # VP ellipsis supplement: runs always, only adds devices not yet found.
        # Needed because spaCy labels the orphaned "off" as adjective/adverb,
        # so the main loop misses "fan" in "turn on the light and off the fan".
        found_devices = {cmd.device for cmd in commands}
        for cmd in self._fallback_particle_scan(doc):
            if cmd.device not in found_devices:
                commands.append(cmd)
                found_devices.add(cmd.device)

        # Last resort: keyword scan for very short / garbled input
        if not commands:
            commands = self._fallback_keyword_scan(doc.text)

        return commands

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_device(self, token) -> Optional[str]:
        """
        Return canonical device name for a token, or None.
        Checks surface form first, then spaCy lemma so plural forms
        like "lights" → "light" and "fans" → "fan" resolve correctly.
        """
        if token.text.lower() in self._devices_lower:
            return token.text.lower()
        if token.lemma_.lower() in self._devices_lower:
            return token.lemma_.lower()
        return None

    def _get_particle(self, token) -> str:
        """
        Return the phrasal-verb particle (on/off) attached to a verb token.
        Handles both proper 'prt' arcs and the common en_core_web_sm
        misparse where on/off is labelled 'prep'.
        """
        for child in token.children:
            if child.dep_ == "prt":
                return child.text.lower()

            if child.dep_ == "prep" and child.text.lower() in self._PARTICLES:
                pobj_tokens = [gc for gc in child.children if gc.dep_ == "pobj"]
                if not pobj_tokens:
                    return child.text.lower()
                if all(gc.text.lower() in self._devices_lower for gc in pobj_tokens):
                    return child.text.lower()
        return ""

    @staticmethod
    def _is_negated(token) -> bool:
        """Check if a verb or its auxiliaries carry a negation (not / n't)."""
        for child in token.children:
            if child.dep_ == "neg":
                return True
            if child.dep_ == "aux":
                for grandchild in child.children:
                    if grandchild.dep_ == "neg":
                        return True
        return False

    @staticmethod
    def _collect_conj_chain(token) -> list:
        """Recursively collect all tokens linked by conj arcs (serial lists)."""
        result = [token]
        for child in token.children:
            if child.dep_ == "conj":
                result.extend(SpacyParser._collect_conj_chain(child))
        return result

    def _find_device_objects(self, verb_token) -> list:
        """
        Walk a verb's children to find device-matching objects.
        Returns raw tokens; caller uses _match_device to get canonical name.
        """
        found = []
        for child in verb_token.children:
            if child.dep_ in ("dobj", "attr", "oprd", "pobj", "nsubjpass"):
                for tok in self._collect_conj_chain(child):
                    if self._match_device(tok):
                        found.append(tok)

            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        for tok in self._collect_conj_chain(grandchild):
                            if self._match_device(tok):
                                found.append(tok)
        return found

    def _fallback_particle_scan(self, doc) -> List[ParsedCommand]:
        """
        Handle VP ellipsis: "turn on the light and off the fan".
        Looks for orphaned on/off near a device, inheriting the last verb seen.
        """
        commands = []
        last_verb = None

        for token in doc:
            if token.pos_ == "VERB":
                last_verb = token
            if (
                token.text.lower() in self._PARTICLES
                and token.dep_ != "prt"
                and last_verb is not None
            ):
                for right in token.rights:
                    for tok in self._collect_conj_chain(right):
                        canonical = self._match_device(tok)
                        if canonical:
                            commands.append(ParsedCommand(
                                device=canonical,
                                action=f"{last_verb.lemma_.lower()} {token.text.lower()}",
                                negated=self._is_negated(last_verb),
                                source_text=doc.text,
                            ))
        return commands

    def _fallback_keyword_scan(self, text: str) -> List[ParsedCommand]:
        """
        Absolute last resort: keyword scan for very short / garbled input.
        Uses lemmatized device forms too (strips trailing 's') as a lightweight
        plural check since spaCy isn't available at this stage.
        """
        text_lower = text.lower()

        action_phrases = {
            "on":  ["turn on", "switch on", "activate", "enable", "start"],
            "off": ["turn off", "switch off", "deactivate", "disable", "stop"],
        }

        commands = []
        added: set = set()

        for state, phrases in action_phrases.items():
            if state not in self._actions:
                continue
            for phrase in phrases:
                if phrase in text_lower:
                    for device in self._devices_lower:
                        # Match exact or naive plural (device + "s")
                        if device in text_lower or (device + "s") in text_lower:
                            key = (device, state)
                            if key not in added:
                                added.add(key)
                                commands.append(ParsedCommand(
                                    device=device,
                                    action=phrase,
                                    negated=False,
                                    source_text=text,
                                ))
        return commands


# ---------------------------------------------------------------------------
# Smoke-test  (python spacy_parser.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_cases = [
        ("turn on the light",                          [("light", "turn on")]),
        ("switch off the fan",                         [("fan", "switch off")]),
        ("turn on the fan and switch off the heater",  [("fan", "turn on"), ("heater", "switch off")]),
        ("turn on the light and fan",                  [("light", "turn on"), ("fan", "turn on")]),
        ("don't turn on the light",                    [("light", "turn on")]),
        ("turn on the light and off the fan",          [("light", "turn on"), ("fan", "turn on")]),
        ("activate the ac",                            [("ac", "activate")]),
        # Plural + filler words
        ("turn on the lights please and also turn off the fan thanks!!",
                                                       [("light", "turn on"), ("fan", "turn off")]),
        ("",                                           []),
    ]

    try:
        parser = SpacyParser()
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("Skipping tests — place config.json next to this file.")
        sys.exit(0)

    passed = failed = 0
    for transcript, expected in test_cases:
        results = parser.parse_as_tuples(transcript)
        ok = set(results) == set(expected)
        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status}  '{transcript}'")
        if not ok:
            print(f"     got:      {results}")
            print(f"     expected: {expected}")

    print(f"\nResults: {passed} passed, {failed} failed")
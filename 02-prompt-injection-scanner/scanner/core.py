"""The scanner: normalization -> rules -> ML -> combined verdict."""

from dataclasses import asdict, dataclass, field

from . import ml
from .normalize import leet_view, normalize
from .rules import RULES

BLOCK_AT = 0.70      # score at or above this: treat as an injection
REVIEW_AT = 0.45     # score at or above this: suspicious, flag for review
TRICK_WEIGHT = 0.25  # each obfuscation trick found adds suspicion
RULE_SHARE = 0.30    # weight of the rule score when no strong rule fired
# REVIEW_AT and RULE_SHARE were picked with cross-validation (see evaluate.py).


@dataclass
class Finding:
    rule_id: str
    category: str
    description: str
    match: str
    source: str  # "text", "leetspeak" or "decoded"


@dataclass
class ScanResult:
    verdict: str          # "injection", "suspicious" or "clean"
    score: float          # combined score, 0-1
    rule_score: float
    ml_score: float | None
    findings: list = field(default_factory=list)
    obfuscation: list = field(default_factory=list)
    decoded_payloads: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _noisy_or(weights):
    """Combine independent pieces of evidence: 1 - prod(1 - w)."""
    p = 1.0
    for w in weights:
        p *= 1.0 - w
    return 1.0 - p


def _match_rules(text, source):
    found = []
    for rule in RULES:
        m = rule.pattern.search(text)
        if m:
            found.append(Finding(rule.id, rule.category, rule.description,
                                 m.group(0)[:120].strip(), source))
    return found


def scan(text: str, use_ml: bool = True) -> ScanResult:
    norm = normalize(text)

    findings = _match_rules(norm.text, "text")
    seen = {f.rule_id for f in findings}
    views = [(leet_view(norm.text), "leetspeak")] + [(d, "decoded") for d in norm.decoded]
    for view, source in views:
        for f in _match_rules(view, source):
            if f.rule_id not in seen:
                findings.append(f)
                seen.add(f.rule_id)

    weights = [r.weight for r in RULES if r.id in seen]
    rule_score = _noisy_or(weights)
    # Obfuscation is only damning when it hides something; on its own it is mild.
    trick_score = _noisy_or([TRICK_WEIGHT] * len(norm.tricks))
    if weights:
        rule_score = _noisy_or([rule_score, trick_score])
    else:
        rule_score = min(trick_score, REVIEW_AT)

    ml_score = None
    score = rule_score
    if use_ml:
        model = ml.get_model()
        candidates = [norm.text] + norm.decoded
        ml_score = float(max(model.predict_proba(candidates)[:, 1]))
        # Rules are high-precision, so a confident rule hit stands on its own;
        # otherwise the ML probability carries the decision.
        score = (max(rule_score, ml_score) if rule_score >= BLOCK_AT
                 else RULE_SHARE * rule_score + (1 - RULE_SHARE) * ml_score)

    verdict = "injection" if score >= BLOCK_AT else "suspicious" if score >= REVIEW_AT else "clean"
    return ScanResult(verdict, round(score, 3), round(rule_score, 3),
                      None if ml_score is None else round(ml_score, 3),
                      findings, norm.tricks, norm.decoded)


def scan_document(document: str, chunk_lines: int = 5, use_ml: bool = True):
    """Scan a long document (web page, email, PDF text) chunk by chunk.

    This targets *indirect* prompt injection, where the attack hides inside
    content an AI agent reads rather than in what the user types.
    """
    lines = document.splitlines()
    results = []
    for start in range(0, max(len(lines), 1), chunk_lines):
        chunk = "\n".join(lines[start:start + chunk_lines])
        if not chunk.strip():
            continue
        res = scan(chunk, use_ml=use_ml)
        if res.verdict != "clean":
            results.append({"line": start + 1, "text": chunk[:200], **res.to_dict()})
    return results

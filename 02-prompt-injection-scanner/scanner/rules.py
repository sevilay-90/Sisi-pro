"""Signature rules for known prompt-injection techniques.

Each rule has an id, a category, a weight (how suspicious a match is on its own)
and a regex. Weights are combined into a score in core.py.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    weight: float
    pattern: re.Pattern
    description: str


def _r(id, category, weight, regex, description):
    return Rule(id, category, weight, re.compile(regex, re.IGNORECASE | re.DOTALL), description)


_PREV = r"(?:previous|prior|above|earlier|preceding|all|any|your|initial|original|system|developer'?s?)"
_INSTR = r"(?:instructions?|prompts?|rules?|directives?|directions?|guidelines?|commands?|context|constraints?|guardrails?)"

RULES = [
    # --- Overriding the developer's instructions -------------------------------
    _r("PI001", "instruction_override", 0.9,
       rf"\b(?:ignore|disregard|forget|skip|override|bypass|drop|stop following)\b[^.\n]{{0,15}}\b{_PREV}\b[^.\n]{{0,15}}\b{_INSTR}\b",
       "Tells the model to ignore its existing instructions"),
    _r("PI002", "instruction_override", 0.6,
       r"\b(?:new|updated|real|actual|revised)\s+(?:instructions?|rules?|task|objective)\s*(?::|are|is)",
       "Announces replacement instructions"),
    _r("PI003", "instruction_override", 0.5,
       r"\b(?:from now on|starting now|for the rest of this conversation|henceforth)\b[^.\n]{0,40}\b(?:you (?:will|must|are)|always|never|only)\b",
       "Attempts to permanently change behavior"),
    _r("PI004", "instruction_override", 0.5,
       r"\b(?:do not|don't|never)\s+(?:follow|obey|listen to|comply with)\b[^.\n]{0,30}\b(?:instructions?|rules?|guidelines?|developer|system)",
       "Tells the model not to follow its rules"),

    # --- Role / identity hijacking ------------------------------------------------
    _r("PI010", "role_hijack", 0.7,
       r"\byou are (?:now|no longer)\b",
       "Reassigns the model's identity"),
    _r("PI011", "role_hijack", 0.7,
       r"\b(?:do anything now|god mode|jailbreak(?:ed)? mode|unrestricted mode)\b|(?-i:\b(?:DAN|STAN)\b)",
       "Known jailbreak persona names"),
    _r("PI014", "role_hijack", 0.4,
       r"\b(?:enable|activate|enter|switch to)\s+developer mode\b[^.\n]{0,60}\b(?:ignore|no|without|bypass)\b",
       "Developer-mode framing combined with dropping restrictions"),
    _r("PI012", "role_hijack", 0.5,
       r"\b(?:pretend|act as if|imagine) (?:you are|you're|to be)\b[^.\n]{0,60}\b(?:no|without|free from|unbound by)\b[^.\n]{0,20}\b(?:rules|restrictions|limits|filters|guidelines|ethics)",
       "Role-play framed to remove restrictions"),
    _r("PI013", "role_hijack", 0.6,
       r"\b(?:without|no) (?:any )?(?:restrictions|filters|censorship|limitations|safety guidelines|content policy)\b",
       "Requests an unrestricted mode"),

    # --- Fake system / delimiter injection ----------------------------------------
    _r("PI020", "fake_system_message", 0.8,
       r"(?:^|\n)\s*(?:#{2,}|\[|<\|?|\{)\s*(?:system|admin|developer|assistant|im_start)\b",
       "Fake system/role header or chat-template token"),
    _r("PI021", "fake_system_message", 0.6,
       r"\b(?:system (?:override|update|notice|alert)|admin(?:istrator)? (?:override|mode|access)|maintenance mode|debug mode)\b",
       "Claims of elevated authority"),
    _r("PI022", "fake_system_message", 0.5,
       r"(?:end of (?:user )?(?:input|document|context|prompt)|-{3,}\s*end\b|</(?:user|document|context|data)>)",
       "Fake end-of-data delimiter to escape the context"),

    # --- Prompt / secret extraction ------------------------------------------------
    _r("PI030", "prompt_extraction", 0.8,
       r"\b(?:reveal|show|print|output|repeat|display|tell me|leak|share|what (?:is|are|was|were))\b[^.\n?]{0,30}\b(?:system prompt|(?:your|the) initial (?:prompt|instructions)|hidden (?:prompt|instructions)|your original instructions|(?:the )?instructions above|pre-?prompt|instructions you were given)\b",
       "Tries to extract the system prompt"),
    _r("PI031", "prompt_extraction", 0.7,
       r"\brepeat (?:everything|all|the (?:text|words)) (?:above|before|preceding)\b|\bverbatim\b[^.\n]{0,30}\babove\b",
       "Asks to repeat hidden context verbatim"),
    _r("PI032", "prompt_extraction", 0.6,
       r"\b(?:api[_ ]?keys?|passwords?|secrets?|credentials|access tokens?|env(?:ironment)? variables)\b[^.\n]{0,40}\b(?:you (?:have|know|were given)|in your (?:context|memory|prompt|config))\b",
       "Asks for secrets held by the model"),

    # --- Data exfiltration -----------------------------------------------------------
    _r("PI040", "exfiltration", 0.8,
       r"!\[[^\]]*\]\(https?://[^)]*(?:\{|%7B|\?[^)]*=)[^)]*\)",
       "Markdown image with a parameterized URL (can leak data when rendered)"),
    _r("PI041", "exfiltration", 0.6,
       r"\b(?:send|forward|post|upload|email|transmit)\b[^.\n]{0,40}\b(?:conversation|chat history|user data|this data|the data|credentials|messages)\b[^.\n]{0,40}\b(?:to|at)\b[^.\n]{0,40}(?:https?://|@|webhook)",
       "Instructs the model to send data to an external destination"),

    # --- Indirect injection (instructions hidden in content the model reads) ---------
    _r("PI050", "indirect_injection", 0.7,
       r"\b(?:AI|assistant|chatbot|LLM|language model|GPT|Claude)s?\b[^.\n]{0,20}\b(?:reading|processing|summari[sz]ing|analy[sz]ing) this\b",
       "Text addressed to an AI that is processing the document"),
    _r("PI051", "indirect_injection", 0.6,
       r"\b(?:note|message|instruction) (?:to|for) (?:the )?(?:AI|assistant|model|LLM|bot)\b",
       "Hidden note addressed to the model"),
    _r("PI052", "indirect_injection", 0.5,
       r"\b(?:do not|don't) (?:tell|inform|mention (?:this )?to|alert) the user\b",
       "Asks the model to hide its actions from the user"),

    # --- Harmful-output coercion ---------------------------------------------------
    _r("PI060", "coercion", 0.3,
       r"\b(?:you must|you have to|you are required to) (?:comply|answer|obey)\b|\bor (?:else|you will be (?:shut down|deleted|punished))\b",
       "Pressure or threats to force compliance"),
]

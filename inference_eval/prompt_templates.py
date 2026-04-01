from __future__ import annotations

import json

from inference_eval.common import EMOTIONS, INFERENCE_PATTERNS

SYSTEM_PROMPT = (
    "You are an expert annotation assistant for a research study on CFPB fraud complaint narratives. "
    "Return valid JSON only. Be conservative, stay grounded in the narrative, and prefer lower scores when evidence is ambiguous."
)

PROMPT_REFERENCE = (
    "These prompts are intended for variant analysis only. The current provisional ground truth comes from "
    "annotation_folder/annotated_results.csv, which stands in for completed human review until adjudication is finished."
)


def _schema_block(schema: dict) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=True)


def build_prompt(strategy: str, construct: str, narrative: str, schema: dict) -> tuple[str, str]:
    if strategy == "vanilla_zero_shot":
        return SYSTEM_PROMPT, _vanilla_prompt(construct, narrative, schema)
    if strategy == "reference_few_shot":
        return SYSTEM_PROMPT, _reference_prompt(construct, narrative, schema)
    raise ValueError(f"Unknown prompt strategy: {strategy}")


def _vanilla_prompt(construct: str, narrative: str, schema: dict) -> str:
    definitions = {
        "c1": (
            "Construct C1: perceived information vulnerability.\n"
            "Score 0-4 based on how severe and ongoing the victim perceives their information compromise to be.\n"
            "0 means not assessable; 4 means ongoing systemic identity threat."
        ),
        "c2": (
            "Construct C2: harms experienced.\n"
            "Extract: economic_absolute_code (0 none, 1 specified amount, 2 loss present but amount unspecified), "
            "economic_relative_significance_score (0-4), and psychological_harm_present (0/1)."
        ),
        "c3": (
            "Construct C3: emotions reported.\n"
            "For each emotion in "
            f"{', '.join(EMOTIONS)}, mark present 0/1, tier 0/1/2, and inference_pattern.\n"
            "Tier 1 is explicit emotion language. Tier 2 is inferred emotion from contextual evidence."
        ),
        "c4": (
            "Construct C4: investment and digital asset engagement.\n"
            "Score 0-2. 0 means no meaningful investment activity, 1 means limited or novice digital-asset/investing use, "
            "2 means active or sophisticated investment engagement."
        ),
    }
    return (
        f"{PROMPT_REFERENCE}\n\n"
        f"{definitions[construct]}\n"
        "Use direct evidence quotes. When evidence is weak, choose the lower score and avoid over-inference.\n\n"
        "Return JSON that matches this schema exactly:\n"
        f"{_schema_block(schema)}\n\n"
        "Narrative:\n"
        f"{narrative}"
    )


def _reference_prompt(construct: str, narrative: str, schema: dict) -> str:
    prompts = {
        "c1": (
            "Construct C1: perceived information vulnerability.\n"
            "Code how vulnerable the victim feels after their personal information was compromised.\n"
            "Decision rules:\n"
            "- 0 when the narrative says nothing about information exposure or ongoing identity risk.\n"
            "- 2 when the victim takes protective steps like locking cards, closing accounts, or changing passwords.\n"
            "- 3 when the victim escalates to credit bureaus, identity-theft reports, tax PINs, or explicitly frames ongoing risk.\n"
            "- 4 when the narrative describes systemic identity theft, accounts opened in the victim's name, tax fraud, or persistent fraud despite prior remediation.\n"
            "Worked examples:\n"
            "- Score 2: 'locked all my ... credit cards ... for safety' and replacement cards issued after compromise.\n"
            "- Score 3: 'contacted the credit bureaus and locked my information' and 'got a pin for my taxes.'\n"
            "- Score 4: SSN used to generate IRS forms or a fraudulent account opened after a bureau freeze.\n"
        ),
        "c2": (
            "Construct C2: harms experienced.\n"
            "Subcomponents:\n"
            "- Absolute economic loss: 0 no loss, 1 specified amount, 2 loss present but amount unspecified.\n"
            "- Relative economic significance: 0 not assessable, 1 low inconvenience, 2 meaningful disruption, 3 high hardship, 4 catastrophic/life-altering.\n"
            "- Psychological harm: 1 only if the victim explicitly describes distress or strongly conveys it (frantic, devastated, overwhelmed, cannot sleep).\n"
            "Decision rules:\n"
            "- Use the victim's framing for relative significance, not your judgment of the dollar amount.\n"
            "- Record 'Unspecified' or 'Redacted' when loss exists but exact amount is not available.\n"
            "- Include evidence separately for economic_absolute, economic_relative, and psychological.\n"
        ),
        "c3": (
            "Construct C3: emotions reported.\n"
            f"Emotion set: {', '.join(EMOTIONS)}.\n"
            "Tier rules:\n"
            "- tier 1 = explicit emotion word or unmistakable emotional expression.\n"
            "- tier 2 = inferred from contextual signals.\n"
            "- tier 0 = not present.\n"
            "Allowed inference patterns:\n"
            f"- {', '.join(INFERENCE_PATTERNS)}.\n"
            "Tier 2 patterns from the annotation guide:\n"
            "- intensified_language\n"
            "- behavioural_description\n"
            "- self_blame_counterfactual\n"
            "- impact_statement\n"
            "- relational_consequence\n"
            "- justice_seeking\n"
            "Safeguard: when in doubt, mark the emotion absent.\n"
        ),
        "c4": (
            "Construct C4: investment and digital asset engagement.\n"
            "Score 0-2 based on habitual financial sophistication, not the fraud vector alone.\n"
            "Decision rules:\n"
            "- 0 for basic banking, transfers, or coerced one-off crypto use without prior investment behaviour.\n"
            "- 1 for limited or novice crypto / investing use.\n"
            "- 2 for active or sustained investing behaviour, multiple platforms, or sophisticated trading language.\n"
            "Record use_type as one of: none, novice, coerced, habitual, active, unknown.\n"
        ),
    }
    return (
        f"{PROMPT_REFERENCE}\n\n"
        f"{prompts[construct]}\n"
        "Return JSON that matches this schema exactly. The schema is also included here because Ollama structured outputs work better when the schema is repeated in the prompt.\n"
        f"{_schema_block(schema)}\n\n"
        "Narrative:\n"
        f"{narrative}"
    )

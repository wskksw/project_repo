from __future__ import annotations

import json
from typing import Any

from inference_eval.common import C0_LABELS, C1_LABELS, C2_REL_LABELS, C4_LABELS, EMOTIONS, INFERENCE_PATTERNS

SYSTEM_PROMPT = (
    "You are an expert annotation assistant for a research study on CFPB fraud complaint narratives. "
    "Return valid JSON only. Follow the schema exactly. Use only evidence grounded in the narrative. "
    "When evidence is ambiguous, choose the lower score and do not speculate. "
    "If the user prompt includes retrieved reference cases, treat them as analogous examples only. "
    "Do not copy their labels unless the target narrative independently supports the same conclusion."
)


def _schema_block(schema: dict) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=True)


def build_prompt(
    strategy: str,
    construct: str,
    narrative: str,
    schema: dict,
    exemplars: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    if strategy == "zero_shot":
        return SYSTEM_PROMPT, _zero_shot_prompt(construct, narrative, schema)
    if strategy == "few_shot":
        return SYSTEM_PROMPT, _few_shot_prompt(construct, narrative, schema)
    if strategy == "retrieval_few_shot":
        return SYSTEM_PROMPT, _retrieval_few_shot_prompt(construct, narrative, schema, exemplars or [])
    raise ValueError(f"Unknown prompt strategy: {strategy}")


def _zero_shot_prompt(construct: str, narrative: str, schema: dict) -> str:
    prompts = {
        "c1": (
            "Call 1 of 4. Code Construct 0 and Construct 1 together.\n"
            "Construct 0: user information leaked. Choose the highest category present.\n"
            f"Allowed labels: {', '.join(C0_LABELS.values())}.\n"
            "0=none, 1=account credentials, 2=personal identification, 3=personal possessions, 4=behavioural characteristics.\n"
            "Construct 1: perceived information vulnerability.\n"
            f"Allowed labels: {', '.join(C1_LABELS.values())}.\n"
            "0=not assessable, 1=low, 2=moderate, 3=high, 4=severe perceived vulnerability.\n"
            "Code C1 from the victim's framing of ongoing risk, not from the objective sensitivity of the data alone.\n"
            "Return evidence for C0 and C1 separately."
        ),
        "c2": (
            "Call 2 of 4. Code Construct 2: harms experienced.\n"
            "Return:\n"
            "- economic_absolute_code: 0=no monetary loss, 1=loss with exact amount, 2=loss present but amount unspecified or redacted.\n"
            "- economic_absolute_amount_usd: null when no loss, otherwise the exact amount string, or 'Unspecified' / 'Redacted'.\n"
            f"- economic_relative_significance_score with labels from: {', '.join(C2_REL_LABELS.values())}.\n"
            "Relative significance is based on how the victim frames the impact of the loss, not your own judgment of the amount."
        ),
        "c3": (
            "Call 3 of 4. Code Construct 3: emotions reported.\n"
            f"Allowed emotion tokens: {', '.join(EMOTIONS)}.\n"
            "Map frustration to anger, fear to anxiety, and keep regret distinct when the narrative is hindsight or self-blame.\n"
            "emotion_count_code: 0=no emotion, 1=one emotion, 2=two or more emotions.\n"
            "evidence_tier: 0 when no emotion, 1 for explicit emotion language, 2 for inferred emotion.\n"
            f"Allowed inference pattern tokens: {', '.join(INFERENCE_PATTERNS)}.\n"
            "If no emotion is present, return emotions=[], evidence_tier=0, inference_patterns=[], evidence_quotes=[]."
        ),
        "c4": (
            "Call 4 of 4. Code Construct 4: financial activities / digital asset engagement.\n"
            f"Allowed labels: {', '.join(C4_LABELS.values())}.\n"
            "0=no digital asset engagement, 1=online transfers and transactions, 2=digital asset engagement.\n"
            "Code the victim's routine or described financial behaviour, not the fraud channel alone."
        ),
    }
    return (
        f"{prompts[construct]}\n\n"
        "Return JSON that matches this schema exactly:\n"
        f"{_schema_block(schema)}\n\n"
        "Narrative:\n"
        f"{narrative}"
    )


def _few_shot_prompt(construct: str, narrative: str, schema: dict) -> str:
    return (
        f"{_few_shot_instructions(construct)}\n\n"
        "Return JSON that matches this schema exactly. Repeat the schema verbatim in your reasoning process and only emit the final JSON object.\n"
        f"{_schema_block(schema)}\n\n"
        "Narrative:\n"
        f"{narrative}"
    )


def _retrieval_few_shot_prompt(
    construct: str,
    narrative: str,
    schema: dict,
    exemplars: list[dict[str, Any]],
) -> str:
    return (
        f"{_few_shot_instructions(construct)}\n\n"
        "Retrieved reference cases:\n"
        f"{_format_exemplars(exemplars)}\n\n"
        "Use the retrieved cases only as annotation references. Base your final labels and evidence only on the target narrative.\n"
        "Return JSON that matches this schema exactly. Repeat the schema verbatim in your reasoning process and only emit the final JSON object.\n"
        f"{_schema_block(schema)}\n\n"
        "Target narrative:\n"
        f"{narrative}"
    )


def _format_exemplars(exemplars: list[dict[str, Any]]) -> str:
    if not exemplars:
        return "- No retrieved reference cases were available."
    rendered = []
    for index, exemplar in enumerate(exemplars, start=1):
        rendered.append(
            f"Example {index} | complaint_id={exemplar['complaint_id']} | similarity={exemplar['similarity_score']:.6f}\n"
            f"Narrative excerpt:\n{exemplar['narrative_excerpt']}\n"
            f"Gold JSON:\n{json.dumps(exemplar['gold_json'], indent=2, ensure_ascii=True)}"
        )
    return "\n\n".join(rendered)


def _few_shot_instructions(construct: str) -> str:
    prompts = {
        "c1": (
            "Call 1 of 4. Code Construct 0 and Construct 1 together.\n"
            "Decision rules for C0:\n"
            "- Use the highest leaked-information category mentioned in the narrative.\n"
            "- account_credentials: usernames, passwords, emails, login access.\n"
            "- personal_identification: name, SSN, date of birth, legal identity data.\n"
            "- personal_possessions: address, phone number, occupation, physical-world possession markers.\n"
            "- behavioural_characteristics: habits, preferences, routines, emotional profile.\n"
            "Decision rules for C1:\n"
            "- 0 when the narrative does not frame information exposure or ongoing risk at all.\n"
            "- 2 when the victim takes protective steps like locking cards, changing passwords, or closing accounts.\n"
            "- 3 when the victim describes sustained risk, bureau freezes, identity-theft reports, or persistent exposure.\n"
            "- 4 when there are cascading identity consequences or ongoing loss of control despite remediation.\n"
            "Worked examples:\n"
            "- Example A: 'I locked all my cards and changed passwords' -> C0 may be account_credentials or personal_identification depending on the breach details, C1=2.\n"
            "- Example B: 'Someone filed taxes with my SSN and opened accounts in my name' -> C0=2 personal_identification, C1=4 severe_perceived_vulnerability.\n"
            "Use schema label tokens exactly."
        ),
        "c2": (
            "Call 2 of 4. Code Construct 2: harms experienced.\n"
            "Decision rules:\n"
            "- economic_absolute_code=0 only when no financial loss is described.\n"
            "- economic_absolute_code=1 when an exact amount is stated. Return the amount as a plain string like 1500.\n"
            "- economic_absolute_code=2 when loss is described but the amount is redacted or missing. Return 'Redacted' or 'Unspecified'.\n"
            "- economic_relative_significance_score=0 when the narrative gives no personal framing of the loss.\n"
            "- Score relative harm from the victim's wording: principled_loss, nuisance_loss, high_impact_loss, life_altering_loss.\n"
            "Worked examples:\n"
            "- 'I paid {$1500.00} and just want the refund I am owed' -> absolute_code=1, amount=1500, relative=1 principled_loss.\n"
            "- 'I lost my retirement savings' -> absolute_code may be 1 or 2 depending on amount visibility, relative=4 life_altering_loss."
        ),
        "c3": (
            "Call 3 of 4. Code Construct 3: emotions reported.\n"
            "Tier rules:\n"
            "- evidence_tier=1 only for direct emotion language such as 'I am embarrassed' or 'I was angry'.\n"
            "- evidence_tier=2 when emotion is inferred from wording or behaviour.\n"
            "Tier-2 inference pattern rules:\n"
            "- intensified_language_and_rhetorical_escalation for accusatory, emphatic, or escalated complaint language.\n"
            "- behavioural_descriptions_of_emotional_states for panic, frantic action, spiralling behaviour, or similar descriptions.\n"
            "- self_blame_or_counterfactual_language for 'I should have known better' or hindsight self-judgment.\n"
            "- impact_statements_on_life_quality for broader life disruption implying sadness or distress.\n"
            "- relational_consequence_descriptions for family/social fallout implying embarrassment.\n"
            "- justice_seeking_language for punitive or lawsuit/prosecution language beyond a routine refund request.\n"
            "Worked examples:\n"
            "- 'It is embarrassing and slandering my professional name' -> emotions=['embarrassment'], evidence_tier=1, inference_patterns=['explicit_emotion'].\n"
            "- 'I may be forced to file a civil lawsuit' -> emotions=['vengeance'], evidence_tier=2, inference_patterns=['justice_seeking_language'].\n"
            "- 'Paypal is being obstinate, hostile and unreasonable' -> emotions=['anger'], evidence_tier=2, inference_patterns=['intensified_language_and_rhetorical_escalation'].\n"
            "If the narrative supports two or more emotions, return all supported canonical tokens."
        ),
        "c4": (
            "Call 4 of 4. Code Construct 4: financial activities.\n"
            "Decision rules:\n"
            "- no_digital_asset_engagement for offline transactions, basic banking only, or no meaningful online financial activity.\n"
            "- online_transfers_and_transactions for one-off or routine online payments, transfers, and digital wallet use.\n"
            "- digital_asset_engagement for sustained investing, crypto activity, multiple online platforms, trading language, or repeated online asset activity.\n"
            "Worked examples:\n"
            "- 'I was told to send money one time from checking and savings' -> usually 1 online_transfers_and_transactions.\n"
            "- 'I traded across multiple crypto platforms and invested significant capital' -> 2 digital_asset_engagement."
        ),
    }
    return prompts[construct]

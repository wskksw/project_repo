#!/usr/bin/env python3
"""
CFPB Fraud Complaint Annotation Pipeline  v2.0
================================================
Annotates CFPB complaint narratives across 4 constructs using the OpenAI API.
Supports single-test mode (one narrative, all constructs) and OpenAI Batch API
for bulk processing.

Constructs:
  C1 — Perceived Information Vulnerability       (scale 0-4)
  C2 — Harms Experienced                         (absolute loss, relative harm, psych harm)
  C3 — Emotions Reported                         (8 emotions × present / tier / inference)
  C4 — Investment & Digital Asset Engagement     (scale 0-2)

Usage:
  python annotation_pipeline.py --test              # run all 4 constructs on first narrative
  python annotation_pipeline.py --batch             # create JSONL, upload, submit batch
  python annotation_pipeline.py --poll  BATCH_ID    # check status of a submitted batch
  python annotation_pipeline.py --retrieve BATCH_ID # download results + write annotated CSV
"""

import os, sys, json, time, argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ============================================================
#  CONFIGURATION  ← set your API key here (or in env var)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

MODEL             = "gpt-5.4-mini"      # reasoning model — verify on your account
REASONING_EFFORT  = "low"       # none | low | medium | high | xhigh
MAX_TOKENS        = 25000          # covers reasoning + output tokens (min 25k recommended)

CSV_INPUT      = "annotation_folder/human_review_sheet_100_samples.csv"
CSV_OUTPUT     = "annotation_folder/annotated_results.csv"
BATCH_JSONL    = "annotation_folder/batch_requests.jsonl"
BATCH_RESULTS  = "annotation_folder/batch_results.jsonl"

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
#  PROMPTS
#  Each construct has its own prompt with multiple worked
#  examples drawn directly from the dataset.
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert annotation assistant for a research study on financial fraud victims. "
    "You read CFPB consumer complaint narratives and extract structured user-model properties. "
    "Always respond with valid JSON only — no markdown fences, no extra text. "
    "Be conservative: when evidence is ambiguous choose the lower score. "
    "Ground every coding decision in direct, verbatim quotes from the narrative."
)

# ------------------------------------------------------------------
# CONSTRUCT 1 — Perceived Information Vulnerability
# ------------------------------------------------------------------
C1_PROMPT = """\
You are coding CONSTRUCT 1: PERCEIVED INFORMATION VULNERABILITY for one CFPB complaint narrative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate how VULNERABLE the victim perceives themselves to be as a result of having their personal
information compromised. This is NOT a checklist of which data types were leaked — it captures
how the victim frames the severity and ongoing impact of information compromise IN THEIR OWN WORDS.
Code based on the victim's language, not your own judgment of which data types are most sensitive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0  Not Assessable           — No textual evidence of how the victim perceives the information
                              compromise. Narrative is purely transactional; no information-
                              exposure concerns are mentioned.

1  Low Perceived Vuln.      — Victim acknowledges information was accessed but frames it as
                              contained or already resolved. No ongoing concern expressed.

2  Moderate Perceived Vuln. — Victim takes protective action (locking cards, changing passwords,
                              closing accounts, requesting replacement cards) indicating concern,
                              but does not describe the compromise as life-altering or express
                              sustained identity-theft fear.

3  High Perceived Vuln.     — Victim explicitly describes ongoing risk. Evidence of sustained
                              concern: contacting credit bureaus, filing identity-theft reports,
                              setting tax PINs, or expressing fear of continued vulnerability
                              that goes beyond immediate account actions.

4  Severe Perceived Vuln.   — Victim describes the compromise as an ongoing, systemic threat.
                              Evidence of cascading consequences: accounts opened in their name,
                              tax fraud using stolen SSN, persistent unauthorized access despite
                              repeated remediation, or explicit language about loss of control
                              over their personal data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Code 0 when the victim says nothing about information exposure at all.
• Behavioural indicators of perceived vulnerability: credit-bureau contact, identity-theft
  filing, tax PIN requests, closing long-held accounts, changing all passwords.
• Cascading identity consequences (tax fraud, new accounts opened, ongoing unauthorized access)
  are strong indicators of 4 even without explicit vulnerability language.
• When uncertain between adjacent scores, choose the lower score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 — Score 0 (Not Assessable)
Narrative excerpt: "I filed a PayPal claim against a seller... I sent the item back to the
seller... There should be no more demands and they should just issue a refund of the $1500 I paid."
Output:
{"perceived_info_vulnerability_score": 0,
 "perceived_info_vulnerability_label": "Not Assessable",
 "evidence_quotes": [],
 "rationale": "Narrative describes a payment dispute with no mention of personal information
   being compromised or any concern about ongoing exposure."}

EXAMPLE 2 — Score 2 (Moderate)
Narrative excerpt: "I locked all my bank of America credit cards on my app as well for safety.
She also stated she would send all new credit cards since these accounts had been compromised."
Output:
{"perceived_info_vulnerability_score": 2,
 "perceived_info_vulnerability_label": "Moderate Perceived Vulnerability",
 "evidence_quotes": ["locked all my bank of America credit cards on my app as well for safety",
                     "send all new credit cards since these accounts had been compromised"],
 "rationale": "Victim locked cards and requested replacements — active but contained protective
   response. No credit-bureau contact or identity-theft filing mentioned."}

EXAMPLE 3 — Score 3 (High)
Narrative excerpt: "Because this person had all of my information, I contacted the credit bureaus
and locked my information. I also got a pin for my taxes. I changed all my passwords."
Output:
{"perceived_info_vulnerability_score": 3,
 "perceived_info_vulnerability_label": "High Perceived Vulnerability",
 "evidence_quotes": ["had all of my information",
                     "contacted the credit bureaus and locked my information",
                     "got a pin for my taxes"],
 "rationale": "Victim escalated beyond immediate card actions to systemic identity safeguarding
   (credit bureaus, tax PIN) — clear signal of sustained ongoing-vulnerability concern."}

EXAMPLE 4 — Score 4 (Severe)
Narrative excerpt: "According to the IRS, my social security number was used to collect money
through Cash App in XXXX. Square, Inc. sent a total of XXXX 1099-Bs to the IRS. I have not ever
had a Cash App or anything similar."
Output:
{"perceived_info_vulnerability_score": 4,
 "perceived_info_vulnerability_label": "Severe Perceived Vulnerability",
 "evidence_quotes": ["my social security number was used to collect money through Cash App",
                     "Square, Inc. sent a total of XXXX 1099-Bs to the IRS",
                     "I have not ever had a Cash App or anything similar"],
 "rationale": "SSN used to open fraudulent accounts and generate IRS tax forms — cascading
   systemic identity fraud with documented government-level consequences."}

EXAMPLE 5 — Score 4 (Severe, persistent)
Narrative excerpt: "I already froze my account for the 3 major credit bureaus approximately 5
months ago, but somehow Cash App allowed an individual to open an account and debit card under my
name without my consent."
Output:
{"perceived_info_vulnerability_score": 4,
 "perceived_info_vulnerability_label": "Severe Perceived Vulnerability",
 "evidence_quotes": ["froze my account for the 3 major credit bureaus approximately 5 months ago",
                     "Cash App allowed an individual to open an account and debit card under my name"],
 "rationale": "Victim took maximum protective action 5 months earlier yet a new fraudulent
   account was still opened — persistent systemic vulnerability despite prior remediation."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return exactly this JSON object (no markdown, no extra keys):
{
  "perceived_info_vulnerability_score": <integer 0-4>,
  "perceived_info_vulnerability_label": "<Not Assessable|Low Perceived Vulnerability|Moderate Perceived Vulnerability|High Perceived Vulnerability|Severe Perceived Vulnerability>",
  "evidence_quotes": ["<verbatim quote 1>", "<verbatim quote 2>"],
  "rationale": "<1-3 sentences grounded in the quoted evidence>"
}

NARRATIVE:
{narrative}
"""

# ------------------------------------------------------------------
# CONSTRUCT 2 — Harms Experienced
# ------------------------------------------------------------------
C2_PROMPT = """\
You are coding CONSTRUCT 2: HARMS EXPERIENCED for one CFPB complaint narrative.
This construct has three independent sub-components. Rate each one separately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2A — ABSOLUTE ECONOMIC LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Record the actual monetary amount lost.

Code 0  Not Present          — No mention of monetary loss.
Code 1  Present – Specified  — Exact dollar amount stated.
Code 2  Present – Unspecified— Financial loss described but no exact figure given.

If code 1, also extract the amount as a string (e.g. "$1,500", "$84,000", "~$100,000").
If the amount appears but is masked (XXXX), write "Redacted".
If multiple amounts, record the total or a brief breakdown string.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2B — RELATIVE ECONOMIC HARM (Perceived Significance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate how SIGNIFICANT the victim perceives their loss relative to their personal financial
situation. The same dollar amount can be catastrophic for one person and trivial for another.
Code based on the victim's OWN framing — not your judgment of the amount.

Score 0  Not Assessable          — Purely factual report; no framing of personal significance.
Score 1  Low                     — Loss framed as inconvenience; frustration about principle,
                                   not financial desperation.
Score 2  Moderate                — Loss is meaningful and caused tangible disruption, but
                                   not described as existentially threatening.
Score 3  High                    — Severe personal financial consequences: hardship, inability
                                   to meet basic needs, or language indicating financial
                                   stability is threatened.
Score 4  Catastrophic            — Life-altering loss: life savings, retirement funds, home
                                   equity, or cascading consequences (eviction, missed mortgage,
                                   negative accounts, fixed-income victim).

Decision rules:
• Look for contextual cues: fixed income, retirement, disability, single-income household.
• Statements like "I can't pay rent," "this was my savings," "I live on SSA benefits."
• Cascading consequences (eviction notice, negative balance, missed mortgage) are strong
  indicators of High or Catastrophic even without explicit framing language.
• If the victim says nothing about relative significance, code 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2C — PSYCHOLOGICAL HARM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Code 0 Not Present — No psychological distress described.
Code 1 Present     — Any negative psychological state caused by the fraud: anxiety, depression,
                     distress, feelings of violation, sleep disruption, or overwhelm.

Code Present only when the victim explicitly describes distress or uses language that strongly
conveys it (e.g., "frantic," "devastated," "suffering," "I can't sleep"). Routine complaint
language alone is NOT sufficient.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 — Abs: 1/$2,100, Rel: 3 (High), Psych: 1
Narrative excerpt: "I ended up losing {$2100.00}. This was just about everything I had at that
time. I was a few days home from a surgery and on major pain killers."
Output:
{"economic_absolute_code": 1,
 "economic_absolute_amount_usd": "$2,100",
 "economic_relative_significance_score": 3,
 "economic_relative_label": "High",
 "psychological_harm_present": 1,
 "evidence_quotes": {
   "economic_absolute": ["losing {$2100.00}"],
   "economic_relative": ["This was just about everything I had at that time"],
   "psychological": ["on major pain killers", "I was a few days home from a surgery"]},
 "rationale": "$2,100 is specified. 'Everything I had' signals high relative significance. The
   victim was medicated and physically vulnerable — psychological harm is strongly implied."}

EXAMPLE 2 — Abs: 1/$1,000, Rel: 4 (Catastrophic), Psych: 1
Narrative excerpt: "I got ta eviction notice I couldnt pay none of my bills and I told them this
they didnt care at all, so now Im just hurt not knowing what to do Im on social security SSA
benefits Im begging you too please help me."
Output:
{"economic_absolute_code": 1,
 "economic_absolute_amount_usd": "$1,000",
 "economic_relative_significance_score": 4,
 "economic_relative_label": "Catastrophic",
 "psychological_harm_present": 1,
 "evidence_quotes": {
   "economic_absolute": ["$1,000"],
   "economic_relative": ["eviction notice", "couldnt pay none of my bills", "on social security SSA benefits"],
   "psychological": ["Im just hurt not knowing what to do", "Im begging you"]},
 "rationale": "SSA fixed income, eviction notice, and inability to pay bills — cascading catastrophic
   consequences. 'Hurt,' 'begging,' and desperation language confirm psychological harm."}

EXAMPLE 3 — Abs: 1/$150,000, Rel: 4 (Catastrophic), Psych: 1
Narrative excerpt: "I suffered a financial loss of approximately {$100,000.00}. I am currently
XXXXXXXX XXXX in service to our country, which makes it even more difficult to address these
matters in person... I expect a full refund."
Output:
{"economic_absolute_code": 1,
 "economic_absolute_amount_usd": "$140,000 (original); ~$100,000 remaining loss",
 "economic_relative_significance_score": 4,
 "economic_relative_label": "Catastrophic",
 "psychological_harm_present": 1,
 "evidence_quotes": {
   "economic_absolute": ["financial loss of approximately {$100,000.00}"],
   "economic_relative": ["makes it even more difficult to address these matters in person",
                         "I am currently XXXXXXXX XXXX in service to our country"],
   "psychological": ["especially distressing"]},
 "rationale": "Six-figure loss for an active service member with limited ability to intervene —
   catastrophic. 'Especially distressing' indicates psychological harm."}

EXAMPLE 4 — Abs: 1/$600, Rel: 2 (Moderate), Psych: 0
Narrative excerpt: "I sent him {$600.00} via PayPal... Losing {$600.00} is a significant
financial setback, and I would really appreciate any help in resolving this issue."
Output:
{"economic_absolute_code": 1,
 "economic_absolute_amount_usd": "$600",
 "economic_relative_significance_score": 2,
 "economic_relative_label": "Moderate",
 "psychological_harm_present": 0,
 "evidence_quotes": {
   "economic_absolute": ["$600.00"],
   "economic_relative": ["a significant financial setback"],
   "psychological": []},
 "rationale": "$600 specified. 'Significant financial setback' indicates meaningful disruption
   but no language of existential hardship. No psychological distress language present."}

EXAMPLE 5 — Abs: 1/$3,700, Rel: 0 (Not Assessable), Psych: 0
Narrative excerpt: "XXXX scammed me and 6 of my friends for a total of {$3700.00} for XXXX
tickets that he never gave us... I would like to request that I be paid back the {$1100.00}
that was wrongfully taken from me."
Output:
{"economic_absolute_code": 1,
 "economic_absolute_amount_usd": "$3,700 total (victim's share: $1,100)",
 "economic_relative_significance_score": 0,
 "economic_relative_label": "Not Assessable",
 "psychological_harm_present": 0,
 "evidence_quotes": {
   "economic_absolute": ["total of {$3700.00}", "be paid back the {$1100.00}"],
   "economic_relative": [],
   "psychological": []},
 "rationale": "Dollar amounts are clear. Victim does not frame the loss relative to their
   personal financial situation. No psychological distress language."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return exactly this JSON object (no markdown, no extra keys):
{
  "economic_absolute_code": <0, 1, or 2>,
  "economic_absolute_amount_usd": "<dollar string, 'Redacted', 'Unspecified', or 'Not Present'>",
  "economic_relative_significance_score": <0, 1, 2, 3, or 4>,
  "economic_relative_label": "<Not Assessable|Low|Moderate|High|Catastrophic>",
  "psychological_harm_present": <0 or 1>,
  "evidence_quotes": {
    "economic_absolute": ["<verbatim quote>"],
    "economic_relative": ["<verbatim quote>"],
    "psychological":     ["<verbatim quote>"]
  },
  "rationale": "<2-4 sentences covering all three sub-components, grounded in quotes>"
}

NARRATIVE:
{narrative}
"""

# ------------------------------------------------------------------
# CONSTRUCT 3 — Emotions Reported
# ------------------------------------------------------------------
C3_PROMPT = """\
You are coding CONSTRUCT 3: EMOTIONS REPORTED for one CFPB complaint narrative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Code the presence of 8 discrete emotions. Use a TWO-TIER approach:

  Tier 1 (Explicit):   Victim uses a direct emotion word ("I was furious," "I felt devastated").
  Tier 2 (Inferred):   Emotion reasonably inferred from an approved language pattern.
                       You MUST name the pattern and quote the evidence.

Approved Tier-2 inference patterns (use these labels exactly in your output):
  "intensified_language"       — Exclamation marks, capitalization, superlatives, or accusatory
                                 verbs conveying emotional arousal beyond routine complaint tone.
  "behavioural_description"    — Actions characteristic of an emotion (e.g., "I was frantic"
                                 signals fear/anxiety without the word "fear").
  "self_blame_counterfactual"  — "I should have..." / "I foolishly..." / "I was so stupid" → shame or regret.
  "impact_on_life_quality"     — How the fraud disrupted daily life, relationships, or wellbeing → sadness.
  "relational_consequence"     — Victim reports how family or others reacted → embarrassment.
  "justice_seeking"            — Legal threats, police/FBI/FTC reports beyond the current complaint → vengeance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMOTION DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
anger         — Hostility, outrage, or frustration at fraudster, institution, or situation.
sadness       — Loss, grief, hopelessness, despair about the fraud outcome.
anxiety  — Worry, dread, or apprehension about future consequences or ongoing vulnerability.
shame         — Personal inadequacy, self-blame, or humiliation about being victimized
                (INTERNAL: "I'm stupid for falling for this").
embarrassment — Social discomfort or concern about others' judgment for being a victim
                (EXTERNAL: "my family is ashamed of me").
disgust       — Revulsion or moral contempt toward the fraudster or the situation.
regret        — Wishes to have acted differently; hindsight-driven distress.
vengeance     — Desire for retaliation, punishment, or justice against the fraudster
                (more than simply requesting a refund).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ANGER is the most common — do NOT code routine complaint language as anger. Look for
  intensified language BEYOND standard complaint framing (e.g., "obstinate, hostile, unreasonable"
  is anger; "I want my money back" alone is not).
• Shame vs. embarrassment: shame is internal self-judgment; embarrassment is social exposure.
  If unclear, code both.
• Vengeance ≠ refund request. "I want my money back" is NOT vengeance. "I will file a civil
  lawsuit" or "I have reported this to the FBI to pursue prosecution" IS vengeance.
• When uncertain, code Not Present (0). Missed emotion < false positive.
• Set tier=0, inference_pattern="none", evidence_quote=null for emotions coded Not Present.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 — anger (Tier 2) + vengeance (Tier 1)
Narrative: "Instead of refunding, Paypal is being obstinate, hostile and unreasonable. Where is
their buyer protection that they tout? I may be forced to file a civil lawsuit for its violations."
Output (emotion subset):
  anger:    present=1, tier=2, pattern="intensified_language",
            evidence="obstinate, hostile and unreasonable"
  vengeance: present=1, tier=1, pattern="none",
             evidence="I may be forced to file a civil lawsuit for its violations"

EXAMPLE 2 — shame + regret (Tier 2, same quote), embarrassment (Tier 2)
Narrative: "I foolishly agreed to help... my whole family is upset that I gotten scammed... I was
not aware that money laundering had gotten so advanced and want to help others avoid the same traps."
Output (emotion subset):
  shame:        present=1, tier=2, pattern="self_blame_counterfactual",
                evidence="I foolishly agreed to help"
  regret:       present=1, tier=2, pattern="self_blame_counterfactual",
                evidence="I was not aware that money laundering had gotten so advanced"
  embarrassment:present=1, tier=2, pattern="relational_consequence",
                evidence="my whole family is upset that I gotten scammed"

EXAMPLE 3 — embarrassment (Tier 1) + sadness (Tier 2)
Narrative: "It is embarrassing and slandering my professional name, not to mention the tens of
people contacting me about bitcoin business and it's not true. I cannot get any real help."
Output (emotion subset):
  embarrassment: present=1, tier=1, pattern="none",
                 evidence="It is embarrassing and slandering my professional name"
  sadness:       present=1, tier=2, pattern="impact_on_life_quality",
                 evidence="tens of people contacting me about bitcoin business and it's not true"

EXAMPLE 4 — anxiety (Tier 2) + vengeance (Tier 1)
Narrative: "I was extremely upset that night, I have a mental illness... What they stole from me
was more than half of my check. I have filed complaints with the FTC and XXXX."
Output (emotion subset):
  anxiety: present=1, tier=2, pattern="behavioural_description",
                evidence="I was extremely upset that night"
  vengeance:    present=1, tier=1, pattern="none",
                evidence="I have filed complaints with the FTC and XXXX"

EXAMPLE 5 — disgust (Tier 2) + vengeance (Tier 1)
Narrative: "I am HORRIFIED to have learned... TD Bank is a convicted criminal bank with convicted
criminal employees. Until these bankers GO TO PRISON this will continue."
Output (emotion subset):
  disgust:   present=1, tier=2, pattern="intensified_language",
             evidence="I am HORRIFIED to have learned... TD Bank is a convicted criminal bank"
  vengeance: present=1, tier=1, pattern="none",
             evidence="Until these bankers GO TO PRISON this will continue"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return exactly this JSON object (no markdown, no extra keys).
For emotions NOT present: set present=0, tier=0, inference_pattern="none", evidence_quote=null.
{
  "emotions": {
    "anger":         {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "sadness":       {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "anxiety":       {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "shame":         {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "embarrassment": {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "disgust":       {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "regret":        {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"},
    "vengeance":     {"present": <0|1>, "tier": <0|1|2>, "inference_pattern": "<label or none>", "evidence_quote": "<verbatim quote or null>"}
  },
  "rationale": "<2-3 sentences explaining the key emotion-coding decisions, grounded in quotes>"
}

NARRATIVE:
{narrative}
"""

# ------------------------------------------------------------------
# CONSTRUCT 4 — Investment and Digital Asset Engagement
# ------------------------------------------------------------------
C4_PROMPT = """\
You are coding CONSTRUCT 4: INVESTMENT AND DIGITAL ASSET ENGAGEMENT for one CFPB complaint narrative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate the victim's level of engagement with investment products, cryptocurrency, or financial
instruments BEYOND basic banking (savings/checking accounts, P2P payment apps such as Zelle,
Venmo, Cash App, or PayPal). This tells us whether the victim had a sophisticated financial
background, dabbled in digital assets, or used only standard banking products.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score 0  No Investment Activity        — Only basic banking described. No investment or
                                         crypto products mentioned.
Score 1  Basic Digital Asset Use       — Victim used cryptocurrency or a trading platform
                                         in a limited or novice capacity (single platform,
                                         small number of transactions, or language indicating
                                         they are new to this).
Score 2  Active Investment Engagement  — Victim describes sustained activity: multiple
                                         platforms, trading strategies (day trading, fractional
                                         shares, staking), significant capital deployed, or
                                         language indicating financial sophistication.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE TYPE — always code this alongside the score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"none"             — Score 0; no investment activity.
"coerced_one_time" — Fraudster directed victim to use crypto/investment platform as part of the
                     scam. No evidence the victim used these products before the fraud.
"habitual"         — Victim was already using investment/crypto products before the fraud.
                     Evidence: prior account history, multiple platforms, fluent financial terms.
"mixed"            — Some prior engagement but fraudster also directed new platform use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Code 0 even if the fraud itself was an "investment scam," as long as the victim describes
  only being directed to use a platform — not that they chose to invest habitually.
• Language signalling sophistication: "day trading," "fractional shares," "margin," "staking,"
  "portfolio," "multiple exchanges." Language signalling novice/coerced: "I was told to invest,"
  "I learned about Coinbase," "I set up an account because they instructed me to."
• Note the coerced/habitual distinction in your reasoning — it changes recovery strategy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 — Score 0 (No Investment Activity)
Narrative: "I filed a PayPal claim against a seller... I sent the item back... they should just
issue a refund of the $1500 I paid for an item that was not as advertised."
Output:
{"investment_engagement_score": 0,
 "investment_engagement_label": "No Investment Activity",
 "use_type": "none",
 "evidence_quotes": [],
 "rationale": "Narrative describes a PayPal goods dispute. No investment or crypto activity mentioned."}

EXAMPLE 2 — Score 1, use_type: habitual
Narrative: "On XX/XX/year>, I made my last authorized transaction on my Coinbase account, which
was a send transaction of XXXX XXXX. Shortly after, a series of unauthorized transactions took
place indicating my account was compromised."
Output:
{"investment_engagement_score": 1,
 "investment_engagement_label": "Basic Digital Asset Use",
 "use_type": "habitual",
 "evidence_quotes": ["my last authorized transaction on my Coinbase account",
                     "a send transaction of XXXX XXXX"],
 "rationale": "Victim had an existing Coinbase account with prior transaction history —
   habitual basic crypto use. No evidence of multi-platform or sophisticated trading."}

EXAMPLE 3 — Score 1, use_type: coerced_one_time
Narrative: "I was misled into authorizing a payment... Crypto.com states that I voluntarily
initiated the transactions to a third-party wallet; however, they failed to implement adequate
security measures... I set up my accounts on these crypto platforms and made payments."
Output:
{"investment_engagement_score": 1,
 "investment_engagement_label": "Basic Digital Asset Use",
 "use_type": "coerced_one_time",
 "evidence_quotes": ["I was misled into authorizing a payment",
                     "I set up my accounts on these crypto platforms and made payments"],
 "rationale": "Victim opened crypto account(s) but only because the fraudster directed them to.
   No prior investment history or sophistication language. Use is coerced, not habitual."}

EXAMPLE 4 — Score 2, use_type: coerced_one_time
Narrative: "Both scams involved exchange platforms like Crypto.com and XXXX. I set up my accounts
on these crypto platforms... There were a total of seven transactions sent from my crypto wallet
to the scammers. These funds were transferred from my bank accounts to my wallet address with
exchange platforms (Exodus and Crypto.com) and then to scammers."
Output:
{"investment_engagement_score": 2,
 "investment_engagement_label": "Active Investment Engagement",
 "use_type": "coerced_one_time",
 "evidence_quotes": ["exchange platforms like Crypto.com and XXXX",
                     "a total of seven transactions sent from my crypto wallet",
                     "exchange platforms (Exodus and Crypto.com)"],
 "rationale": "Victim engaged with multiple crypto platforms (Crypto.com, Exodus, XXXX) across
   seven transactions — substantive multi-platform engagement, even though entirely fraud-directed.
   Coded score 2 because of multi-platform use, but use_type is coerced (no prior investment language)."}

EXAMPLE 5 — Score 2, use_type: habitual
Narrative: "I had started to study bigger stock trades... I planned day trading in near future,
I transferred {$5,000} in funds... I traded Bitcoin, various stocks both whole shares and
fractions of shares."
Output:
{"investment_engagement_score": 2,
 "investment_engagement_label": "Active Investment Engagement",
 "use_type": "habitual",
 "evidence_quotes": ["study bigger stock trades",
                     "planned day trading",
                     "traded Bitcoin, various stocks both whole shares and fractions of shares"],
 "rationale": "Victim independently pursued multi-asset investment (Bitcoin + equities + fractional
   shares + day trading strategy). This is sustained, self-directed investment activity — habitual."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return exactly this JSON object (no markdown, no extra keys):
{
  "investment_engagement_score": <0, 1, or 2>,
  "investment_engagement_label": "<No Investment Activity|Basic Digital Asset Use|Active Investment Engagement>",
  "use_type": "<none|coerced_one_time|habitual|mixed>",
  "evidence_quotes": ["<verbatim quote 1>", "<verbatim quote 2>"],
  "rationale": "<2-3 sentences explaining the score and use_type, grounded in quotes>"
}

NARRATIVE:
{narrative}
"""

# Map construct key → prompt template
CONSTRUCT_PROMPTS = {
    "c1": C1_PROMPT,
    "c2": C2_PROMPT,
    "c3": C3_PROMPT,
    "c4": C4_PROMPT,
}


# ============================================================
#  CORE HELPERS
# ============================================================

def build_messages(prompt_template: str, narrative: str) -> list[dict]:
    """Build the messages list for a single API call."""
    user_content = prompt_template.replace("{narrative}", narrative.strip())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def call_single_construct(prompt_template: str, narrative: str) -> dict:
    """Call the Responses API synchronously for one construct. Used in test mode."""
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=build_messages(prompt_template, narrative),
        text={"format": {"type": "json_object"}},
        max_output_tokens=MAX_TOKENS,
    )
    return json.loads(response.output_text)


# ============================================================
#  TEST MODE  (single narrative, all 4 constructs)
# ============================================================

def test_on_first_narrative(df: pd.DataFrame) -> None:
    """Run all 4 constructs on the first non-empty narrative and pretty-print results."""
    valid = df[df["Consumer complaint narrative"].notna() &
               (df["Consumer complaint narrative"].str.strip() != "")]
    if valid.empty:
        print("No non-empty narratives found.")
        return

    row = valid.iloc[0]
    complaint_id = row["Complaint ID"]
    narrative    = str(row["Consumer complaint narrative"])

    print(f"\n{'='*70}")
    print(f"  TEST RUN — Complaint ID: {complaint_id}")
    print(f"  Narrative preview: {narrative[:300].strip()}...")
    print(f"{'='*70}\n")

    all_results = {}
    for key, prompt in CONSTRUCT_PROMPTS.items():
        print(f"  ▶  Running {key.upper()}...", end="  ", flush=True)
        try:
            result = call_single_construct(prompt, narrative)
            all_results[key] = result
            print("OK")
            print(json.dumps(result, indent=4))
        except Exception as exc:
            print(f"ERROR: {exc}")
            all_results[key] = {"error": str(exc)}
        print()

    return all_results


# ============================================================
#  BATCH MODE  (create JSONL → upload → submit)
# ============================================================

def create_batch_jsonl(df: pd.DataFrame, output_path: str) -> int:
    """Write one JSONL batch-request file covering all constructs × all rows.

    Custom ID format: {complaint_id}-{construct}
    e.g.  11380835-c1, 11380835-c2, 11380835-c3, 11380835-c4
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(output_path, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            narrative = row.get("Consumer complaint narrative", "")
            if pd.isna(narrative) or not str(narrative).strip():
                continue
            narrative = str(narrative).strip()
            complaint_id = str(row["Complaint ID"])

            for construct, prompt_template in CONSTRUCT_PROMPTS.items():
                request = {
                    "custom_id": f"{complaint_id}-{construct}",
                    "method":    "POST",
                    "url":       "/v1/responses",
                    "body": {
                        "model":             MODEL,
                        "reasoning":         {"effort": REASONING_EFFORT},
                        "input":             build_messages(prompt_template, narrative),
                        "text":              {"format": {"type": "json_object"}},
                        "max_output_tokens": MAX_TOKENS,
                    },
                }
                fh.write(json.dumps(request, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Batch JSONL written: {output_path}  ({n_written} requests)")
    return n_written


def submit_batch(jsonl_path: str) -> str:
    """Upload JSONL file and submit batch job. Returns batch ID."""
    print("Uploading batch file to OpenAI Files API...")
    with open(jsonl_path, "rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")
    print(f"  File ID:  {upload.id}")

    print("Submitting batch...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": "CFPB annotation pipeline v2"},
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status:   {batch.status}")
    print(f"\nSave this ID → run with --retrieve {batch.id} when complete.")
    return batch.id


# ============================================================
#  RETRIEVE / POLL
# ============================================================

def poll_batch_status(batch_id: str, poll_interval: int = 30) -> object:
    """Block until batch completes (or fails). Returns final batch object."""
    print(f"Polling batch {batch_id}  (interval {poll_interval}s)...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [{batch.status}]  "
              f"completed={counts.completed}  failed={counts.failed}  total={counts.total}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch
        time.sleep(poll_interval)


def download_results(batch_id: str, output_path: str) -> str | None:
    """Retrieve completed batch and write raw JSONL results to disk."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch status is '{batch.status}' — not ready to download.")
        return None
    if not batch.output_file_id:
        print("No output_file_id on batch. Nothing to download.")
        return None

    print(f"Downloading results (file {batch.output_file_id})...")
    content = client.files.content(batch.output_file_id)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(content.content)
    print(f"Results saved: {output_path}")
    return output_path


# ============================================================
#  PARSE RESULTS → DataFrame
# ============================================================

def _safe_join(lst: list) -> str:
    """Join a list of strings with ' | ' separator, filtering None/empty."""
    return " | ".join(str(x) for x in lst if x)


def parse_and_merge(results_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Parse batch output JSONL and write AI annotations into df.

    Fills:  c1_*, c2_*, c3_*, c4_*  columns
    Leaves: *_human_override  columns blank  (for reviewer)
    Sets:   review_status = "pending"
    """
    # ---- Load raw results ----
    parsed: dict[str, dict] = {}
    errors = 0
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            cid  = item["custom_id"]
            if item.get("error"):
                print(f"  API error for {cid}: {item['error']}")
                errors += 1
                continue
            try:
                body    = item["response"]["body"]
                # Responses API: find the message item in the output array
                content = None
                for output_item in body.get("output", []):
                    if output_item.get("type") == "message":
                        for c in output_item.get("content", []):
                            if c.get("type") == "output_text":
                                content = c["text"]
                                break
                        break
                if content is None:
                    raise ValueError("No output_text found in response output")
                parsed[cid] = json.loads(content)
            except Exception as exc:
                print(f"  Parse error for {cid}: {exc}")
                errors += 1

    print(f"Loaded {len(parsed)} results  ({errors} errors)")

    df = df.copy()

    # Empty annotation columns are read as float64 by pandas; cast to object
    # so string values (labels, JSON, quotes) can be written without dtype errors.
    annotation_cols = [
        "c1_score", "c1_label", "c1_evidence_quotes",
        "c2_abs_code", "c2_abs_amount", "c2_rel_score", "c2_rel_label",
        "c2_psych_harm", "c2_evidence_quotes",
        "c3_emotions_json", "c3_tier_json", "c3_inference_pattern_json", "c3_evidence_quotes",
        "c4_score", "c4_label", "c4_use_type", "c4_evidence_quotes",
        "review_status",
    ]
    for col in annotation_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    for idx, row in df.iterrows():
        complaint_id = str(row["Complaint ID"])

        # ---- C1 ----
        r = parsed.get(f"{complaint_id}-c1", {})
        if r and "error" not in r:
            df.at[idx, "c1_score"]           = r.get("perceived_info_vulnerability_score", "")
            df.at[idx, "c1_label"]           = r.get("perceived_info_vulnerability_label", "")
            df.at[idx, "c1_evidence_quotes"] = _safe_join(r.get("evidence_quotes", []))

        # ---- C2 ----
        r = parsed.get(f"{complaint_id}-c2", {})
        if r and "error" not in r:
            df.at[idx, "c2_abs_code"]        = r.get("economic_absolute_code", "")
            df.at[idx, "c2_abs_amount"]      = r.get("economic_absolute_amount_usd", "")
            df.at[idx, "c2_rel_score"]       = r.get("economic_relative_significance_score", "")
            df.at[idx, "c2_rel_label"]       = r.get("economic_relative_label", "")
            df.at[idx, "c2_psych_harm"]      = r.get("psychological_harm_present", "")
            eq = r.get("evidence_quotes", {})
            all_quotes = (
                eq.get("economic_absolute", []) +
                eq.get("economic_relative", []) +
                eq.get("psychological", [])
            )
            df.at[idx, "c2_evidence_quotes"] = _safe_join(all_quotes)

        # ---- C3 ----
        r = parsed.get(f"{complaint_id}-c3", {})
        if r and "error" not in r:
            # Filter to only dict-valued entries; some responses embed
            # 'rationale' as a key inside emotions instead of at the top level.
            emotions = {k: v for k, v in r.get("emotions", {}).items()
                        if isinstance(v, dict)}
            df.at[idx, "c3_emotions_json"] = json.dumps(
                {k: v.get("present", 0) for k, v in emotions.items()}
            )
            df.at[idx, "c3_tier_json"] = json.dumps(
                {k: v.get("tier", 0) for k, v in emotions.items()}
            )
            df.at[idx, "c3_inference_pattern_json"] = json.dumps(
                {k: v.get("inference_pattern", "none") for k, v in emotions.items()}
            )
            quotes = [
                v.get("evidence_quote") for v in emotions.values()
                if v.get("evidence_quote")
            ]
            df.at[idx, "c3_evidence_quotes"] = _safe_join(quotes)

        # ---- C4 ----
        r = parsed.get(f"{complaint_id}-c4", {})
        if r and "error" not in r:
            df.at[idx, "c4_score"]           = r.get("investment_engagement_score", "")
            df.at[idx, "c4_label"]           = r.get("investment_engagement_label", "")
            df.at[idx, "c4_use_type"]        = r.get("use_type", "")
            df.at[idx, "c4_evidence_quotes"] = _safe_join(r.get("evidence_quotes", []))

        # Mark as pending human review; leave *_human_override blank
        df.at[idx, "review_status"] = "pending"

    return df


# ============================================================
#  MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFPB Annotation Pipeline — test, batch, poll, retrieve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotation_pipeline.py --test
  python annotation_pipeline.py --batch
  python annotation_pipeline.py --poll    batch_abc123
  python annotation_pipeline.py --retrieve batch_abc123
        """,
    )
    parser.add_argument("--test",     action="store_true",
                        help="Run all 4 constructs on the first narrative only")
    parser.add_argument("--batch",    action="store_true",
                        help="Create JSONL, upload to OpenAI, and submit batch job")
    parser.add_argument("--poll",     metavar="BATCH_ID",
                        help="Poll status of a submitted batch")
    parser.add_argument("--retrieve", metavar="BATCH_ID",
                        help="Download completed batch results and write annotated CSV")
    args = parser.parse_args()

    if not any([args.test, args.batch, args.poll, args.retrieve]):
        parser.print_help()
        sys.exit(0)

    # ---- Load input ----
    print(f"Loading: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)
    print(f"  {len(df)} rows loaded.")

    # ---- TEST ----
    if args.test:
        test_on_first_narrative(df)

    # ---- BATCH SUBMIT ----
    elif args.batch:
        n = create_batch_jsonl(df, BATCH_JSONL)
        if n == 0:
            print("No valid narratives found. Aborting.")
            sys.exit(1)
        submit_batch(BATCH_JSONL)

    # ---- POLL ----
    elif args.poll:
        poll_batch_status(args.poll)

    # ---- RETRIEVE ----
    elif args.retrieve:
        # Poll until complete, then download
        final_batch = poll_batch_status(args.retrieve, poll_interval=10)
        results_path = download_results(args.retrieve, BATCH_RESULTS)
        if results_path is None:
            sys.exit(1)
        df_annotated = parse_and_merge(results_path, df)
        df_annotated.to_csv(CSV_OUTPUT, index=False)
        print(f"\nAnnotated CSV saved: {CSV_OUTPUT}")
        print("Human reviewers should open this file and fill in the *_human_override columns.")
        print("Set review_status = 'reviewed' when complete.")


if __name__ == "__main__":
    main()

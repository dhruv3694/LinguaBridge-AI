"""Prompt Engineering Module (notebook cell 16). No heavy deps."""
from __future__ import annotations

SUPPORTED_LANGUAGES = {"Hindi": "Hindi", "Marathi": "Marathi"}
SUPPORTED_STYLES = {
    "Neutral": "Translate naturally and accurately.",
    "Formal": "Use formal, respectful and grammatically correct language.",
    "Informal": "Use casual and friendly everyday language.",
    "Conversational": "Use natural spoken language as in everyday conversation.",
    "Academic": "Use precise academic vocabulary and a scholarly register.",
    "Professional": "Use polished professional language suitable for workplace communication.",
}
SUPPORTED_TECHNIQUES = ["zero_shot", "few_shot", "role_based"]

def validate_inputs(text, target_language, style="Neutral"):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError("Target language must be Hindi or Marathi.")
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"Unsupported style. Choose from: {list(SUPPORTED_STYLES)}")
    return True

def build_zero_shot_prompt(text, target_language, style="Neutral"):
    validate_inputs(text, target_language, style)
    return f"""
Translate the following English text into {target_language}.

Requirements:
- Preserve the original meaning.
- Do not omit important information.
- Do not add explanations.
- Maintain grammatical correctness.
- {SUPPORTED_STYLES[style]}

English text:
{text}

Return only the translated text.
""".strip()

def build_few_shot_prompt(text, target_language, style="Neutral", examples=None):
    validate_inputs(text, target_language, style)
    if examples is None:
        examples = {
            "Hindi": [("Good morning.", "सुप्रभात।"), ("How are you?", "आप कैसे हैं?")],
            "Marathi": [("Good morning.", "शुभ सकाळ."), ("How are you?", "तुम्ही कसे आहात?")],
        }
    prompt = f"""
Translate English into {target_language}.

Follow the examples below for translation quality and style.

Style: {SUPPORTED_STYLES[style]}

"""
    for source, translation in examples.get(target_language, []):
        prompt += f"English: {source}\n{target_language}: {translation}\n\n"
    prompt += f"""
Now translate this sentence:

English: {text}
{target_language}:
"""
    return prompt.strip()

def build_role_based_prompt(text, target_language, style="Neutral"):
    validate_inputs(text, target_language, style)
    return f"""
You are an expert English-to-{target_language} translator.

Your responsibilities:
1. Produce an accurate translation.
2. Preserve meaning and context.
3. Use culturally appropriate expressions.
4. Avoid unnecessary additions or omissions.
5. Follow the requested language register.

Required style:
{SUPPORTED_STYLES[style]}

Text to translate:
{text}

Return only the {target_language} translation.
""".strip()

def build_translation_prompt(text, target_language="Hindi", style="Neutral", technique="zero_shot"):
    builders = {"zero_shot": build_zero_shot_prompt, "few_shot": build_few_shot_prompt, "role_based": build_role_based_prompt}
    if technique not in builders:
        raise ValueError(f"Unsupported technique. Choose from: {SUPPORTED_TECHNIQUES}")
    return builders[technique](text, target_language, style)

def compare_prompt_techniques(text, target_language="Hindi", style="Formal"):
    return {technique: build_translation_prompt(text=text, target_language=target_language, style=style, technique=technique)
            for technique in SUPPORTED_TECHNIQUES}

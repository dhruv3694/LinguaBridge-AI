"""
Unit Tests for Phase 2: Prompt Engineering Module.
Validates reusable zero-shot, few-shot, role-based, and tone-controlled prompts.
"""

import pytest
from modules.prompt_engineering import (
    FEW_SHOT_EXEMPLARS,
    SUPPORTED_LANGUAGES,
    TranslationTechnique,
    TranslationTone,
    build_chat_messages,
    build_translation_prompt,
    normalize_language,
    normalize_technique,
    normalize_tone,
)


# ============================================================================
# 1. Input Validation and Normalization Tests
# ============================================================================

def test_empty_input_rejection():
    """Verify that empty strings raise a ValueError."""
    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        build_translation_prompt(text="", target_language="Hindi")


def test_whitespace_input_rejection():
    """Verify that whitespace-only strings raise a ValueError."""
    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        build_translation_prompt(text="   \n\t  ", target_language="Marathi")


def test_non_string_input_rejection():
    """Verify that non-string inputs raise a TypeError."""
    with pytest.raises(TypeError, match="valid string"):
        build_translation_prompt(text=12345, target_language="Hindi")  # type: ignore


def test_unsupported_language_rejection():
    """Verify that unsupported languages raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported target language"):
        build_translation_prompt(text="Hello world", target_language="French")


def test_unsupported_technique_rejection():
    """Verify that unsupported prompting techniques raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported technique"):
        build_translation_prompt(
            text="Hello world",
            target_language="Hindi",
            technique="chain-of-thought",
        )


def test_unsupported_tone_rejection():
    """Verify that unsupported tones raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported tone"):
        build_translation_prompt(
            text="Hello world",
            target_language="Marathi",
            tone="sarcastic",
        )


def test_language_alias_normalization():
    """Verify that aliases like 'hi', 'hin', 'mr' normalize correctly."""
    assert normalize_language("hi") == "Hindi"
    assert normalize_language("HINDI") == "Hindi"
    assert normalize_language("mr") == "Marathi"
    assert normalize_language("marathi") == "Marathi"


# ============================================================================
# 2. Reusable Prompt Template Constraints Verification
# ============================================================================

@pytest.mark.parametrize("target_lang", ["Hindi", "Marathi"])
def test_all_prompts_contain_required_specifications(target_lang):
    """
    Every generated prompt must specify:
    - Translator role
    - Source language (English)
    - Target language
    - Original input
    - Desired tone
    - Accuracy / meaning preservation constraints
    - Output format instructions
    - Anti-hallucination constraint
    """
    text = "Artificial intelligence will transform healthcare delivery."
    prompt = build_translation_prompt(
        text=text,
        target_language=target_lang,
        technique="zero-shot",
        tone="formal",
    )

    # 1. Translator role
    assert "translator" in prompt.lower()
    # 2. Source language
    assert "English" in prompt
    # 3. Target language & native name
    assert target_lang in prompt
    assert SUPPORTED_LANGUAGES[target_lang]["native_name"] in prompt
    # 4. Original input text
    assert text in prompt
    # 5. Tone directive
    assert "TONE REQUIREMENT" in prompt
    # 6. Accuracy constraints
    assert "PRESERVE MEANING" in prompt
    # 7. Output format instructions
    assert "OUTPUT FORMAT INTEGRITY" in prompt
    assert "Output ONLY the raw" in prompt
    # 8. Anti-hallucination instruction
    assert "ZERO INVENTIONS" in prompt
    assert "Do NOT invent" in prompt


# ============================================================================
# 3. Prompting Techniques Verification
# ============================================================================

def test_zero_shot_prompt_has_no_in_context_examples():
    """Zero-shot prompt should not contain few-shot in-context examples."""
    prompt = build_translation_prompt(
        text="The team accomplished their annual milestones.",
        target_language="Hindi",
        technique="zero-shot",
    )
    assert "### FEW-SHOT IN-CONTEXT EXAMPLES:" not in prompt


def test_few_shot_prompt_contains_multilingual_examples():
    """Few-shot prompt must contain English, Hindi, and Marathi exemplars."""
    prompt_hi = build_translation_prompt(
        text="The conference begins tomorrow morning.",
        target_language="Hindi",
        technique="few-shot",
        tone="formal",
    )
    assert "### FEW-SHOT IN-CONTEXT EXAMPLES:" in prompt_hi
    assert "Source English:" in prompt_hi
    assert "Target Hindi:" in prompt_hi

    prompt_mr = build_translation_prompt(
        text="The conference begins tomorrow morning.",
        target_language="Marathi",
        technique="few-shot",
        tone="formal",
    )
    assert "Target Marathi:" in prompt_mr

    # Ensure all exemplars in dataset contain all three languages
    for exemplar in FEW_SHOT_EXEMPLARS:
        assert "english" in exemplar and len(exemplar["english"]) > 0
        assert "hindi" in exemplar and len(exemplar["hindi"]) > 0
        assert "marathi" in exemplar and len(exemplar["marathi"]) > 0


def test_role_based_prompt_contains_senior_persona():
    """Role-based prompt must define an authoritative senior linguist persona."""
    prompt = build_translation_prompt(
        text="The legal contract was executed successfully.",
        target_language="Hindi",
        technique="role-based",
        tone="formal",
    )
    assert "Senior Indic Linguist and Master Translator" in prompt
    assert "20 years of experience" in prompt
    assert "syntactic" in prompt.lower() or "syntax" in prompt.lower()


# ============================================================================
# 4. Tone and Register Variations Verification
# ============================================================================

def test_formal_tone_specifications():
    """Formal prompt must specify polite honorifics and respectful pronouns."""
    prompt_hi = build_translation_prompt("Please submit the report.", "Hindi", tone="formal")
    assert "Strictly FORMAL and polite" in prompt_hi
    assert "आप (aap)" in prompt_hi

    prompt_mr = build_translation_prompt("Please submit the report.", "Marathi", tone="formal")
    assert "Strictly FORMAL and polite" in prompt_mr
    assert "आपण (aapan)" in prompt_mr or "तुम्ही (tumhi)" in prompt_mr


def test_informal_tone_specifications():
    """Informal prompt must specify casual, intimate phrasing and informal pronouns."""
    prompt_hi = build_translation_prompt("What are you up to?", "Hindi", tone="informal")
    assert "INFORMAL and intimate" in prompt_hi
    assert "तुम (tum)" in prompt_hi or "तू (tu)" in prompt_hi

    prompt_mr = build_translation_prompt("What are you up to?", "Marathi", tone="informal")
    assert "INFORMAL and intimate" in prompt_mr
    assert "तू (tu)" in prompt_mr


def test_academic_tone_specifications():
    """Academic prompt must mandate scholarly, objective domain terminology."""
    prompt = build_translation_prompt(
        "Quantum superposition was verified experimentally.",
        "Hindi",
        tone="academic",
    )
    assert "ACADEMIC and scholarly" in prompt
    assert "objective" in prompt.lower()
    assert "Devanagari" in prompt


def test_conversational_tone_specifications():
    """Conversational prompt must mandate natural spoken dialogue cadence."""
    prompt = build_translation_prompt(
        "I'm really excited for the weekend trip!",
        "Marathi",
        tone="conversational",
    )
    assert "NATURAL and CONVERSATIONAL" in prompt
    assert "everyday speech" in prompt.lower()


def test_professional_tone_specifications():
    """Professional prompt must mandate corporate and administrative register."""
    prompt = build_translation_prompt(
        "Please submit the budget forecast by tomorrow.",
        "Hindi",
        tone="professional",
    )
    assert "PROFESSIONAL and corporate" in prompt
    assert "workplace inflection" in prompt.lower() or "business-appropriate" in prompt.lower()


# ============================================================================
# 5. Chat Template Message Formatting Verification
# ============================================================================

def test_build_chat_messages_schema():
    """Verify chat message dictionaries compatible with tokenizer.apply_chat_template()."""
    messages = build_chat_messages(
        text="Welcome to the laboratory.",
        target_language="Hindi",
        technique="zero-shot",
        tone="formal",
    )
    assert isinstance(messages, list)
    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Welcome to the laboratory." in messages[-1]["content"]


def test_build_chat_messages_few_shot_turns():
    """Verify few-shot chat messages contain assistant demonstration turns."""
    messages = build_chat_messages(
        text="Hello world",
        target_language="Marathi",
        technique="few-shot",
        tone="formal",
    )
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "assistant" in roles
    assert "user" in roles

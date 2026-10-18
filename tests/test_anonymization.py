"""Anonymization in Hebrew: prefixes, names that are also words, two people with one name.

The cases here are the reason anonymization has its own task. Each test is a sentence a
counselor could really say in a staff meeting.
"""

from datetime import UTC, datetime
from itertools import product

import pytest
from fakes import World

from pacer.ai.anonymization import (
    PREFIX_LETTERS,
    AnonymizedText,
    Anonymizer,
    Person,
    PersonKind,
    Roster,
    deanonymize,
    load_roster,
    tokens_in,
)
from pacer.domain import TraineeStatus

T, S = PersonKind.TRAINEE, PersonKind.STAFF
PEOPLE = [
    Person(T, 7, "יעל", "מזרחי"),
    Person(T, 8, "יעל", "כהן"),  # a second Yael: "יעל" alone is ambiguous
    Person(T, 3, "שחר", "ששון"),  # שחר is also "dawn"
    Person(T, 4, "טל", "בר"),  # טל is also "dew", בר also a word
    Person(T, 5, "הדר", "גבאי"),  # starts with ה, a prefix letter
    Person(T, 6, "מאיה", "אוחיון"),  # starts with מ, a prefix letter
    Person(T, 9, "בן", "גולן"),  # בן is also "son"
    Person(S, 2, "נועה", "לוי"),
    Person(S, 10, "עומר", "כהן"),  # shares the last name כהן with trainee 8
]


@pytest.fixture(scope="module")
def anonymizer() -> Anonymizer:
    return Anonymizer(Roster(PEOPLE))


def anon(anonymizer: Anonymizer, text: str) -> str:
    return anonymizer.anonymize(text).text


# --- the basics ---------------------------------------------------------------------------


def test_full_first_and_last_name_all_become_the_same_token(anonymizer):
    assert anon(anonymizer, "מאיה אוחיון רצה 5 ק״מ") == "[TRAINEE_6] רצה 5 ק״מ"
    assert anon(anonymizer, "מאיה רצה") == "[TRAINEE_6] רצה"
    assert anon(anonymizer, "אוחיון רצה") == "[TRAINEE_6] רצה"


def test_staff_and_trainees_get_different_kinds_of_tokens(anonymizer):
    assert anon(anonymizer, "נועה תדבר עם מאיה") == "[STAFF_2] תדבר עם [TRAINEE_6]"


def test_text_without_names_is_unchanged(anonymizer):
    text = "הקבוצה יצאה להתנדבות בצפון, וחזרה מאוחר. Next meeting on Sunday."
    assert anon(anonymizer, text) == text


# --- Hebrew prefix letters -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("דיברתי עם מאיה ולמאיה", "דיברתי עם [TRAINEE_6] ול[TRAINEE_6]"),
        ("ומאיה הגיעה", "ו[TRAINEE_6] הגיעה"),
        ("כשמאיה סיימה", "כש[TRAINEE_6] סיימה"),
        ("מנועה קיבלתי", "מ[STAFF_2] קיבלתי"),
        ("וכשהדר באה", "וכש[TRAINEE_5] באה"),
    ],
    ids=["ל", "ו", "כש", "מ", "וכש"],
)
def test_prefix_letters_stay_and_the_name_becomes_a_token(anonymizer, text, expected):
    assert anon(anonymizer, text) == expected


def test_four_prefix_letters_are_beyond_hebrew_and_not_matched(anonymizer):
    """Hebrew stacks at most three prefix letters ("וכש"). A documented, tested limit."""
    assert anon(anonymizer, "ושכשהדר באה") == "ושכשהדר באה"


def test_a_name_that_starts_with_a_prefix_letter_is_kept_whole(anonymizer):
    """הדר is a name, not ה + דר. מאיה is a name, not מ + איה."""
    assert anon(anonymizer, "הדר ומאיה") == "[TRAINEE_5] ו[TRAINEE_6]"
    assert anon(anonymizer, "להדר") == "ל[TRAINEE_5]"


def test_every_name_with_every_prefix_combination_is_caught(anonymizer):
    """Exhaustive: each surface form, behind every prefix of up to three letters."""
    prefixes = [""] + ["".join(p) for n in (1, 2, 3) for p in product(PREFIX_LETTERS, repeat=n)]
    roster = Roster(PEOPLE)
    for form, prefix in product(roster.forms, prefixes):
        result = anonymizer.anonymize(f"אמרתי {prefix}{form} משהו")
        assert not anonymizer.contains_known_name(result.text), (prefix, form)


# --- names that are also words: masked, but as the WORD (COMMON_WORD_NAMES) -------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("קמנו בשחר לריצה", "קמנו ב[NAME_1] לריצה"),  # dawn
        ("היה טל על הדשא", "היה [NAME_1] על הדשא"),  # dew
        ("הבן של השכנים", "ה[NAME_1] של השכנים"),  # son
    ],
)
def test_names_that_are_also_words_are_masked_with_a_neutral_token(anonymizer, text, expected):
    """Still the safe direction (the word never leaves as-is), but the token points to no one."""
    safe = anonymizer.anonymize(text)
    assert safe.text == expected
    assert deanonymize(safe.text, safe) == text  # back to the word, not to a person
    assert not safe.people


def test_kvar_does_not_come_back_as_yael_bar():
    """The case from the real run on 2026-10-15: "כבר" is כ + בר, and בר is a trainee's last name.
    Before the fix, Claude's summary came back as "שלחתי אותו כיעל בר"."""
    anonymizer = Anonymizer(Roster([Person(T, 14, "יעל", "בר"), Person(S, 1, "רונית", "שמעוני")]))
    safe = anonymizer.anonymize("רונית: שלחתי אותו כבר אתמול. יעל בר תבדוק.")
    assert safe.text == "[STAFF_1]: שלחתי אותו כ[NAME_1] אתמול. [TRAINEE_14] תבדוק."  # full name: her
    answer = "[STAFF_1] שלחה את הלו״ז כ[NAME_1] אתמול, ו[TRAINEE_14] תבדוק."
    assert deanonymize(answer, safe) == "רונית שמעוני שלחה את הלו״ז כבר אתמול, ויעל בר תבדוק."


def test_a_common_word_name_is_the_person_when_the_context_says_who():
    shahar = Person(S, 11, "שחר", "מימון")
    roster = Roster([*PEOPLE, shahar])  # two people named שחר: trainee 3 and counselor 11
    anonymizer = Anonymizer(roster)
    assert anonymizer.anonymize("שחר, תתאם", known=[shahar]).text == "[STAFF_11], תתאם"  # he is at the meeting
    assert anonymizer.anonymize("שחר סיפר", prefer=[PEOPLE[2]]).text == "[TRAINEE_3] סיפר"  # a talk with him
    # both in context: still neutral, never a guess
    assert anonymizer.anonymize("שחר", known=[shahar], prefer=[PEOPLE[2]]).text == "[NAME_1]"


def test_a_full_name_with_a_common_word_is_always_the_person(anonymizer):
    assert anon(anonymizer, "טל בר הגיעה, ושחר ששון גם") == "[TRAINEE_4] הגיעה, ו[TRAINEE_3] גם"


@pytest.mark.parametrize("word", ["שחררו", "טלפון", "מטלטל", "בנייה", "ברזל", "הדרכה"])
def test_names_inside_longer_words_are_not_touched(anonymizer, word):
    """שחר inside שחררו, טל inside טלפון: not a name, not masked."""
    assert anon(anonymizer, f"אמרו {word} היום") == f"אמרו {word} היום"


# --- two people, one name ----------------------------------------------------------------------


def test_full_names_resolve_two_yaels_to_the_right_people(anonymizer):
    assert anon(anonymizer, "יעל מזרחי ויעל כהן") == "[TRAINEE_7] ו[TRAINEE_8]"


def test_an_ambiguous_first_name_gets_a_neutral_token(anonymizer):
    result = anonymizer.anonymize("יעל איחרה, ואז יעל התנצלה")
    assert result.text == "[NAME_1] איחרה, ואז [NAME_1] התנצלה"  # same word, same token
    assert result.unresolved == {"[NAME_1]": "יעל"}
    assert result.people == {}  # no one was guessed


def test_a_shared_last_name_is_ambiguous_too(anonymizer):
    """כהן is both staff member 10 and trainee 8."""
    assert anon(anonymizer, "כהן אמר") == "[NAME_1] אמר"


def test_spaces_and_line_breaks_inside_a_full_name(anonymizer):
    assert anon(anonymizer, "יעל   מזרחי\nויעל\nכהן") == "[TRAINEE_7]\nו[TRAINEE_8]"


# --- back to names, on this computer -------------------------------------------------------------


def test_deanonymize_restores_full_names_and_ambiguous_words(anonymizer):
    source = anonymizer.anonymize("נועה ויעל ידברו עם מאיה")
    model_answer = "[STAFF_2] תתאם שיחה עם [TRAINEE_6]. כדאי לשתף את [NAME_1]."
    assert deanonymize(model_answer, source) == "נועה לוי תתאם שיחה עם מאיה אוחיון. כדאי לשתף את יעל."


def test_a_token_the_model_invented_is_left_for_validation(anonymizer):
    source = anonymizer.anonymize("נועה")
    assert deanonymize("[STAFF_2] ו-[STAFF_99]", source) == "נועה לוי ו-[STAFF_99]"
    assert tokens_in("[STAFF_2] ו-[STAFF_99] ו[NAME_3]") == ["[STAFF_2]", "[STAFF_99]", "[NAME_3]"]


def test_tokens_map_back_to_people(anonymizer):
    source = anonymizer.anonymize("נועה ומאיה")
    assert source.person_for("[STAFF_2]").full_name == "נועה לוי"
    assert source.person_for("[STAFF_99]") is None


# --- the type is the proof --------------------------------------------------------------------------


def test_anonymized_text_cannot_be_made_by_hand():
    with pytest.raises(TypeError, match="only by Anonymizer"):
        AnonymizedText(text="נועה", people={}, unresolved={}, _key=object())


def test_anonymized_text_never_shows_names_in_its_repr(anonymizer):
    result = anonymizer.anonymize("נועה ומאיה")
    assert "נועה" not in repr(result) and "מאיה" not in repr(result)
    assert repr(result).startswith("AnonymizedText(")


def test_an_empty_roster_changes_nothing():
    assert Anonymizer(Roster([])).anonymize("נועה ומאיה").text == "נועה ומאיה"


# --- the roster comes from the database ------------------------------------------------------------


def test_roster_has_every_staff_member_and_every_trainee_in_any_status():
    world = World()
    world.trainees.update(world.yael.with_status(TraineeStatus.LEFT))
    roster = load_roster(world.staff, world.trainees)

    kinds = {(p.kind, p.full_name) for p in roster.people}
    assert (S, "Noa Levi") in kinds and (S, "Ronit Manager") in kinds
    assert (T, "Yael Mizrahi") in kinds  # left the academy, still protected


def test_the_whole_demo_academy_leaves_no_name_behind():
    """All 35 trainees and 5 staff of the demo academy, in one long text, with prefixes."""
    from test_demo_academy import fake_repos

    from pacer.demo import build_academy

    repos = fake_repos()
    build_academy(repos, as_of=datetime(2026, 10, 14, tzinfo=UTC))
    anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))

    mentions = [f"ו{t.first_name} דיבר עם {t.full_name}" for t in repos.trainees.list_all()]
    mentions += [f"ל{s.full_name}" for s in repos.staff.list_all()]
    result = anonymizer.anonymize(". ".join(mentions))

    assert not anonymizer.contains_known_name(result.text)
    assert len(result.people) + len(result.unresolved) > 30


def test_a_whole_name_wins_over_prefix_plus_a_shorter_name():
    """מאור and אור are both common names. "מאור" is Maor, not מ + Or: the prefix match must be lazy."""
    anonymizer = Anonymizer(Roster([Person(T, 1, "מאור", "דהן"), Person(T, 2, "אור", "שלו")]))
    # אור is also a word ("light"), so alone it is neutral; the point is that מאור is not split
    assert anonymizer.anonymize("מאור ואור").text == "[TRAINEE_1] ו[NAME_1]"
    assert anonymizer.anonymize("למאור").text == "ל[TRAINEE_1]"


def test_known_people_map_back_even_when_the_text_never_names_them(anonymizer):
    noa = Person(S, 2, "נועה", "לוי")
    safe = anonymizer.anonymize("צריך לתאם הסעה", known=[noa])
    assert safe.text == "צריך לתאם הסעה"
    assert deanonymize("[STAFF_2] תתאם", safe) == "נועה לוי תתאם"


def test_the_roster_finds_a_person_by_kind_and_id():
    roster = Roster(PEOPLE)
    assert roster.find(S, 2).first_name == "נועה"
    assert roster.find(T, 2) is None  # id 2 is a staff member, not a trainee


# --- spelling variants and names of several words (found in an independent review) --------------


def test_each_word_of_a_two_word_first_name_is_her_too():
    sara_lea = Person(T, 40, "שרה לאה", "פרידמן")
    anonymizer = Anonymizer(Roster([sara_lea]))
    assert anonymize_text(anonymizer, "שרה הגיעה, ולאה נשארה. שרה לאה פרידמן") == (
        "[TRAINEE_40] הגיעה, ו[TRAINEE_40] נשארה. [TRAINEE_40]"
    )


def test_latin_names_match_in_any_case():
    dana = Person(S, 41, "Dana", "Levi")
    anonymizer = Anonymizer(Roster([dana]))
    assert anonymize_text(anonymizer, "dana and DANA LEVI met Dana") == "[STAFF_41] and [STAFF_41] met [STAFF_41]"


@pytest.mark.parametrize("written", ["ג'ני", "ג׳ני", "ג’ני", "ג`ני"])
def test_apostrophe_and_geresh_are_the_same_name(written):
    jenny = Person(T, 42, "ג׳ני", "אלמוג")  # the roster spells it with a geresh
    assert anonymize_text(Anonymizer(Roster([jenny])), f"{written} הגיעה") == "[TRAINEE_42] הגיעה"


def test_hyphen_and_maqaf_are_the_same_name():
    person = Person(T, 43, "עדי", "בר-און")
    anonymizer = Anonymizer(Roster([person]))
    assert anonymize_text(anonymizer, "משפחת בר־און ועדי בר-און") == "משפחת [TRAINEE_43] ו[TRAINEE_43]"


def test_an_ambiguous_name_in_two_cases_gets_one_token_and_comes_back_as_written():
    roster = Roster([Person(S, 44, "Dana", "Levi"), Person(S, 45, "Dana", "Katz")])
    safe = Anonymizer(roster).anonymize("Dana said dana")
    assert safe.text == "[NAME_1] said [NAME_1]"
    assert deanonymize("[NAME_1]", safe) == "Dana"


def anonymize_text(anonymizer: Anonymizer, text: str) -> str:
    return anonymizer.anonymize(text).text

def test_who_the_text_is_about_resolves_an_ambiguous_first_name(anonymizer):
    yael_mizrahi = Person(T, 7, "יעל", "מזרחי")
    about_her = anonymizer.anonymize("יעל סיפרה שהיא רצה", prefer=[yael_mizrahi])
    no_context = anonymizer.anonymize("יעל סיפרה שהיא רצה")
    assert about_her.text == "[TRAINEE_7] סיפרה שהיא רצה" and about_her.unresolved == {}
    assert no_context.text == "[NAME_1] סיפרה שהיא רצה"


def test_preferring_someone_does_not_affect_names_that_are_not_theirs(anonymizer):
    shahar = Person(T, 3, "שחר", "ששון")
    safe = anonymizer.anonymize("כהן ושחר", prefer=[shahar])
    assert safe.text == "[NAME_1] ו[TRAINEE_3]"  # כהן fits two others: still ambiguous

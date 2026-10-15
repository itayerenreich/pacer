"""A sample personal talk, in Hebrew, with a trainee of the demo academy. Invented.

מאיה גולן (trainee 22, counselor איתי ביטון, id 5). Her goals in the demo: 62 lead a social evening
(0%), 63 run 5 km (25%), 64 lead a full day for the group (50%), 65 keep a journal twice a week (45%).

Written to exercise the pipeline: clear progress on one goal, a goal that did not move, things that
worked and did not, an open follow-up, an ambiguous name (רותם: two trainees), and health details
that must never become a note (headaches, a doctor).
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SampleTalk:
    trainee_id: int
    counselor_id: int
    held_on: date
    notes: str


MAYA = SampleTalk(
    trainee_id=22,
    counselor_id=5,
    held_on=date(2026, 10, 13),
    notes="""\
שיחה אישית עם מאיה גולן.
מאיה סיפרה שהשבוע רצה 3 ק"מ ברצף, פעמיים, לעומת 2 ק"מ בחודש שעבר. היא מרוצה מההתקדמות.
ביומן האישי היא כותבת רק פעם בשבוע. אמרה שבערבים היא עייפה מדי. סיכמנו שתנסה לכתוב בבוקר, לפני המסדר.
מה שעבד: כשקבענו יעד קטן ומדיד (עוד חצי ק"מ כל שבועיים) היא התמידה.
מה שפחות עבד: תזכורות בקבוצת הווטסאפ, היא מתעלמת מהן.
היא רוצה לדבר שוב על הובלת היום הקבוצתי, כי היא מתלבטת אם להוביל אותו יחד עם רותם. סיכמנו שאחזור אליה בשבוע הבא.
היא גם סיפרה שיש לה כאבי ראש בזמן האחרון ושהיא הולכת לרופא ביום חמישי.
""",
)

SAMPLE_TALKS: dict[str, SampleTalk] = {"maya": MAYA}

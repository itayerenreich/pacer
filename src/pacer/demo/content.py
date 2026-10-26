"""The Hebrew content of the demo academy: names, goals, notes, intake answers.

Everything here is invented. Following the privacy decisions in the data model:
no ID numbers, addresses or medical details, and nothing about family income or
neighborhoods. Background comes only from the kind of intake questionnaire a
trainee fills in themselves.
"""

from pacer.domain import GoalCategory, ResponsibilityArea

COHORT_NAME = "מחזור י״ב"

MANAGER_NAME = "רונית שמעוני"

# (name, area). Two counselors per group: the first two lead group A, the last two group B.
COUNSELORS: list[tuple[str, ResponsibilityArea]] = [
    ("נועה לוי", ResponsibilityArea.VOLUNTEERING),
    ("עומר כהן", ResponsibilityArea.ARMY_PREP),
    ("שירה אברהם", ResponsibilityArea.CONTENT),
    ("איתי ביטון", ResponsibilityArea.LOGISTICS),
]

GROUP_NAMES = ["קבוצת ארז", "קבוצת אלון"]

FIRST_NAMES = [
    "יעל",
    "אורי",
    "תמר",
    "נועם",
    "מאיה",
    "איתמר",
    "שחר",
    "רוני",
    "הילה",
    "עידו",
    "ליה",
    "דניאל",
    "אביגיל",
    "יונתן",
    "שקד",
    "אריאל",
    "נגה",
    "עמית",
    "טליה",
    "גיא",
    "אדל",
    "יהונתן",
    "רותם",
    "אלמוג",
    "מיכל",
    "בן",
    "הדר",
    "ינאי",
    "שני",
    "איתן",
    "עדי",
    "נדב",
    "אגם",
    "עומרי",
    "ליאור",
    "זיו",
    "אסף",
    "רעות",
    "יואב",
    "מור",
]

LAST_NAMES = [
    "מזרחי",
    "פרץ",
    "ביטון",
    "דהן",
    "אברהם",
    "פרידמן",
    "אזולאי",
    "כץ",
    "שפירא",
    "חדד",
    "גבאי",
    "מלכה",
    "סויסה",
    "רוזן",
    "קליין",
    "עמר",
    "אוחיון",
    "וקנין",
    "בן דוד",
    "טל",
    "שושן",
    "נחום",
    "יוסף",
    "גולן",
    "הררי",
    "לנדאו",
    "בר",
    "אשכנזי",
    "ששון",
    "רביבו",
]

GOALS: dict[GoalCategory, list[str]] = {
    GoalCategory.PERSONAL: [
        "לקום בזמן בכל בוקר בלי תזכורת",
        "לקרוא ספר אחד בחודש",
        "לנהל יומן אישי ולכתוב בו פעמיים בשבוע",
        "לסדר את סדר היום כך שיהיה זמן ללימוד אישי",
    ],
    GoalCategory.SOCIAL: [
        "להכיר לעומק כל אחד מחברי הקבוצה",
        "להוביל ערב חברתי לכל המחזור",
        "לפתור מחלוקות בשיחה ישירה ולא מאחורי הגב",
        "להשתתף באופן פעיל בשיחות הקבוצה",
    ],
    GoalCategory.VALUES: [
        "להתנדב שעתיים בשבוע במרכז הקהילתי",
        "להוביל דיון ערכי בקבוצה על אחריות אישית",
        "לבנות ולהעביר פעילות לבני נוער בשכונה",
    ],
    GoalCategory.PHYSICAL: [
        "לרוץ 5 ק״מ ברצף",
        "להגיע ל-15 עליות מתח",
        "לשפר את זמן ריצת ה-3,000 בדקה",
        "להתאמן שלוש פעמים בשבוע גם בסופי שבוע",
    ],
    GoalCategory.LEADERSHIP: [
        "להוביל פעילות של יום שלם לקבוצה",
        "לקחת אחריות על תורנות המטבח לשבוע",
        "להיות אחראי/ת מחזור בטיול השנתי",
        "לתת משוב בונה לחבר בכל שבוע",
    ],
    GoalCategory.ARMY_PREP: [
        "להתכונן לגיבוש ולעבור את כל שלביו",
        "ללמוד על מסלולי השירות ולבחור שלושה מועדפים",
        "לשפר כושר לקראת יום המא״ה",
    ],
}

# Progress notes by stage. A counselor writes one of these on each update.
NOTES_EARLY = [
    "התחלנו לעבוד על זה בשיחה האישית",
    "בנינו יחד תוכנית עם צעדים קטנים",
    "יש מוטיבציה, צריך להפוך אותה להרגל",
    "צעד ראשון בוצע, ממשיכים",
]
NOTES_MIDDLE = [
    "התקדמות יפה השבוע, עקביות טובה",
    "היה שבוע קשה, אבל לא ויתר/ה",
    "רואים שיפור ברור לעומת תחילת החודש",
    "ממשיכים בקצב, קבענו בדיקה בעוד שבועיים",
    "קיבל/ה משוב מהקבוצה ולקח/ה אותו ברצינות",
]
NOTES_DONE = [
    "הושג! חגגנו בשיחת הקבוצה",
    "המטרה הושגה, נבחר יחד את הצעד הבא",
    "עמד/ה ביעד לפני הזמן",
]
NOTE_DROPPED = "הוחלט יחד להוריד את המטרה ולהתמקד במטרות אחרות"

INTAKE_QUESTIONS = {
    "למה בחרת במכינה?": [
        "רציתי שנה של צמיחה אישית לפני הצבא",
        "אחי הגדול היה במכינה וסיפר שזו השנה הכי משמעותית שלו",
        "חיפשתי מקום שבו אלמד להוביל אנשים",
        "רציתי להכיר אנשים שונים ממני ולתרום לקהילה",
        "הרגשתי שאני עוד לא מוכן/ה לצבא ורציתי להתכונן",
    ],
    "מה החוזקה שלך?": [
        "אני יודע/ת להקשיב",
        "אני לא מוותר/ת בקלות",
        "אני מצחיק/ה ומחבר/ת בין אנשים",
        "אני מסודר/ת ואחראי/ת",
        "אני סקרן/ית ואוהב/ת ללמוד",
    ],
    "במה תרצה/י להשתפר השנה?": [
        "לדבר מול קבוצה בלי להתבייש",
        "להתמיד בכושר",
        "לקבל ביקורת בלי להיעלב",
        "לנהל את הזמן שלי",
        "לבקש עזרה כשאני צריך/ה",
    ],
    "תחביבים": ["גיטרה", "כדורסל", "צילום", "בישול", "טיולים", "קריאה", "ריצה", "ציור"],
}

# Approved long-term notes (TraineeNote) for a few trainees: what a counselor learned about them, as
# if approved from personal talks. The counselor coach (stage 5) reads these, so that its advice can
# build on what worked and what did not with THIS trainee. Educational observations only, never health.
# (kind, text, days before the demo's "now")
TRAINEE_NOTES: dict[str, list[tuple[str, str, int]]] = {
    "נדב הררי": [
        ("observation", "בשיחות אחד על אחד הוא פתוח ומדבר הרבה, בשיחות הקבוצה כמעט שותק", 30),
        ("did_not_work", "כשפניתי אליו בשמו מול כל הקבוצה ושאלתי מה דעתו, הוא הסמיק ולא ענה", 24),
        ("worked", "כשביקשתי ממנו מראש להכין נקודה אחת לשיחה, הוא אמר אותה ואפילו הוסיף משהו", 12),
        ("follow_up", "לבדוק אם ירצה להוביל חלק קטן מהשיחה הקבוצתית הבאה", 5),
    ],
    "בן שפירא": [
        ("observation", "מאחר לפעילויות הבוקר בערך פעמיים בשבוע, בעיקר אחרי ערבים חברתיים", 26),
        ("did_not_work", "תזכורות חוזרות ממני הפכו למאבק כוח, והוא התחיל לאחר יותר", 18),
        ("worked", "מאז שקיבל תפקיד של אחראי ציוד לבוקר, הגיע בזמן כמעט כל יום", 8),
    ],
    "טליה חדד": [
        ("observation", "נכנסת לוויכוחים חריפים עם שותפה לחדר, בעיקר סביב סדר וניקיון", 21),
        ("worked", "שיחה משותפת עם שתיהן אחרי שנרגעו, עם כללים לחדר שהן ניסחו בעצמן", 14),
        ("follow_up", "לבדוק אחרי החופשה אם הכללים בחדר עדיין מחזיקים", 3),
    ],
    "איתן רוזן": [
        ("observation", "בהפסקות מצחיק ומרכז את כולם, בשיחות ערכיות שותק או מתבדח", 20),
        ("worked", "שאלה פתוחה בכתב יום לפני השיחה: הגיע עם תשובה רצינית ושיתף", 9),
    ],
    "אגם לנדאו": [
        ("did_not_work", "השוואה לחניכים אחרים באימוני הכושר רק הורידה לה את המוטיבציה", 17),
        ("worked", "טבלת מעקב אישית מול התוצאות של עצמה מתחילת השנה", 10),
        ("observation", "לא מדברת בשיחות הקבוצה, אבל כותבת הרבה ובעומק במשימות בכתב", 6),
    ],
}

# The weekly template (decision 2026-10-26). (weekdays, start, end, title, per group?)
# Weekdays in Python's numbering: 6 Sunday, 0 Monday, 1 Tuesday, 2 Wednesday, 3 Thursday, 4 Friday.
SUNDAY_TO_THURSDAY = (6, 0, 1, 2, 3)
SCHEDULE: list[tuple[tuple[int, ...], str, str, str, bool]] = [
    (SUNDAY_TO_THURSDAY, "06:45", "07:45", "ספורט בוקר", False),
    (SUNDAY_TO_THURSDAY, "08:30", "10:00", "שיעור עיון", False),
    ((6, 1, 3), "10:30", "12:30", "סדנה", False),
    ((0,), "16:00", "17:30", "שיחת קבוצה", True),  # each group has its own
    ((2,), "14:00", "17:00", "התנדבות בקהילה", False),
    ((3,), "20:00", "21:30", "ערב מחזור", False),
    ((4,), "08:00", "11:00", "תורנויות", False),
]
# How likely each trainee is to be absent (not excused), by default and in some slots. Patterns, not
# noise: they match the approved notes (Ben is late after social evenings; Agam's fitness frustration).
ABSENCE_DEFAULT = 0.05
EXCUSED = 0.03
ABSENCE_BY_TRAINEE: dict[str, dict[str | None, float]] = {
    "בן שפירא": {"ספורט בוקר": 0.45},
    "אגם לנדאו": {"ספורט בוקר": 0.25},
    "נדב נחום": {None: 0.25},  # every slot: a trainee whose goals have not started either
    "איתמר סויסה": {"שיעור עיון": 0.3},
}
ATTENDANCE_WEEKS = 3  # weeks of attendance before the demo's "now", plus the current week so far

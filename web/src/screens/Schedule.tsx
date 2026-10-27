import { useEffect, useState } from "react";

import { api, type Mark, type Session, type Sheet } from "../api";
import type { Academy } from "../App";
import { addDays, DAY_NAMES, shortDate, sundayOf, today } from "../dates";

const MARKS: { value: Mark; label: string }[] = [
  { value: "present", label: "הגיע" },
  { value: "absent", label: "לא הגיע" },
  { value: "excused", label: "מוצדק" },
];

export function Schedule({ academy }: { academy: Academy }) {
  const [sunday, setSunday] = useState(() => sundayOf(today()));
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    setSessions(null);
    setProblem(null);
    api
      .week(sunday, academy.cohortId)
      .then(setSessions)
      .catch((error: Error) => setProblem(error.message));
  }, [sunday, academy.cohortId]);

  const days = Array.from({ length: 6 }, (_, i) => addDays(sunday, i)); // Sunday to Friday
  const now = today();

  return (
    <>
      <header>
        <h1>לו״ז ונוכחות</h1>
        <p className="quiet">בחרו משבצת כדי לסמן מי הגיע. מסמנים מהיום של המשבצת והלאה.</p>
      </header>
      <div className="week-nav">
        <button className="link" type="button" onClick={() => setSunday(addDays(sunday, -7))}>
          השבוע הקודם
        </button>
        <strong>
          {shortDate(sunday)} עד {shortDate(addDays(sunday, 5))}
        </strong>
        <button className="link" type="button" onClick={() => setSunday(addDays(sunday, 7))}>
          השבוע הבא
        </button>
        {sunday !== sundayOf(now) && (
          <button className="link" type="button" onClick={() => setSunday(sundayOf(now))}>
            השבוע
          </button>
        )}
      </div>
      {problem && <p className="error">{problem}</p>}
      {sessions && sessions.length === 0 && <p className="quiet">השבוע הזה עוד לא פורסם.</p>}
      {sessions && sessions.length > 0 && (
        <div className="week">
          {days.map((day, i) => (
            <div key={day} className={day === now ? "day today" : "day"}>
              <h3>
                {DAY_NAMES[i]} {shortDate(day)}
              </h3>
              {sessions
                .filter((s) => s.on === day)
                .sort((a, b) => a.start.localeCompare(b.start))
                .map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={s.cancelled ? "slot cancelled" : "slot"}
                    aria-pressed={open === s.id}
                    onClick={() => setOpen(s.id)}
                  >
                    <time>
                      <bdi dir="ltr">
                        {s.start.slice(0, 5)}–{s.end.slice(0, 5)}
                      </bdi>
                    </time>
                    {s.title}
                    {s.group_id !== null && <span className="quiet"> ({academy.groups.get(s.group_id)?.name})</span>}
                  </button>
                ))}
            </div>
          ))}
        </div>
      )}
      {open !== null && <AttendanceSheet key={open} sessionId={open} />}
    </>
  );
}

function AttendanceSheet({ sessionId }: { sessionId: number }) {
  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [marks, setMarks] = useState<Record<number, Mark>>({});
  const [problem, setProblem] = useState<string | null>(null);
  const [state, setState] = useState<"idle" | "saving" | "saved">("idle");

  useEffect(() => {
    api
      .sheet(sessionId)
      .then((s) => {
        setSheet(s);
        setMarks(Object.fromEntries(s.trainees.filter((t) => t.status).map((t) => [t.trainee_id, t.status!])));
      })
      .catch((error: Error) => setProblem(error.message));
  }, [sessionId]);

  if (problem && !sheet) return <p className="error sheet">{problem}</p>;
  if (!sheet) return <p className="quiet sheet">טוען...</p>;

  const session = sheet.session;
  const changed = sheet.trainees.filter((t) => marks[t.trainee_id] && marks[t.trainee_id] !== t.status);
  const everyonePresent = () =>
    setMarks((m) => ({ ...Object.fromEntries(sheet.trainees.map((t) => [t.trainee_id, "present" as Mark])), ...m }));

  const save = async () => {
    setState("saving");
    setProblem(null);
    try {
      const saved = await api.mark(
        session.id,
        changed.map((t) => ({ trainee_id: t.trainee_id, status: marks[t.trainee_id] })),
      );
      setSheet(saved);
      setState("saved");
    } catch (error) {
      setProblem((error as Error).message);
      setState("idle");
    }
  };

  return (
    <section className="sheet">
      <h2>
        {session.title}, {session.weekday} {shortDate(session.on)}
      </h2>
      {session.cancelled ? (
        <p className="quiet">המשבצת בוטלה: אין בה נוכחות.</p>
      ) : (
        <>
          <p>
            <button className="link" type="button" onClick={everyonePresent}>
              לסמן את כל מי שלא סומן כ״הגיע״
            </button>
          </p>
          <ul className="ruled">
            {sheet.trainees.map((t) => (
              <li key={t.trainee_id}>
                <span>{t.name}</span>
                <span className="marks" role="group" aria-label={`נוכחות של ${t.name}`}>
                  {MARKS.map((m) => (
                    <button
                      key={m.value}
                      type="button"
                      className={m.value}
                      aria-pressed={marks[t.trainee_id] === m.value}
                      onClick={() => {
                        setMarks((all) => ({ ...all, [t.trainee_id]: m.value }));
                        setState("idle");
                      }}
                    >
                      {m.label}
                    </button>
                  ))}
                </span>
              </li>
            ))}
          </ul>
          <p style={{ marginTop: "1rem" }}>
            <button className="button" type="button" disabled={changed.length === 0 || state === "saving"} onClick={() => void save()}>
              {state === "saving" ? "שומר..." : `שמירת הנוכחות${changed.length ? ` (${changed.length})` : ""}`}
            </button>{" "}
            {state === "saved" && changed.length === 0 && <span className="saved">הנוכחות נשמרה</span>}
          </p>
          {problem && <p className="error">{problem}</p>}
        </>
      )}
    </section>
  );
}

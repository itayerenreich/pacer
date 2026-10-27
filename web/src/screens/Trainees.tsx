import { useEffect, useMemo, useState } from "react";

import { api, type AttendanceSummary, type Goal, type Note, type Task, type Trainee } from "../api";
import type { Academy } from "../App";
import { deadlineLabel, today } from "../dates";

const GOAL_STATUS: Record<Goal["status"], string> = {
  not_started: "טרם התחילה",
  in_progress: "בתהליך",
  achieved: "הושגה",
  dropped: "ירדה",
};

const NOTE_KIND: Record<Note["kind"], string> = {
  worked: "עבד",
  did_not_work: "לא עבד",
  observation: "תצפית",
  follow_up: "למעקב",
};

const STATUS: Record<Trainee["status"], string> = {
  active: "",
  on_leave: "בחופשה",
  left: "עזב/ה",
  graduated: "סיים/ה",
};

export function Trainees({ academy }: { academy: Academy }) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<number | null>(null);

  const byGroup = useMemo(() => {
    const words = query.trim();
    const all = [...academy.trainees.values()].filter((t) => !words || `${t.first_name} ${t.last_name}`.includes(words));
    const groups = new Map<number, Trainee[]>();
    for (const t of all.sort((a, b) => a.first_name.localeCompare(b.first_name, "he"))) {
      groups.set(t.group_id, [...(groups.get(t.group_id) ?? []), t]);
    }
    return [...groups.entries()];
  }, [academy.trainees, query]);

  return (
    <>
      <header>
        <h1>חניכים</h1>
        <p className="quiet">
          {academy.me.role === "manager" ? "כל החניכים במכינה." : "החניכים בקבוצות שלך."} התיק האישי נפתח רק באישור, ולא
          מוצג כאן.
        </p>
      </header>
      <div className="two-pane">
        <div className="roster">
          <label htmlFor="find" className="visually-hidden">
            חיפוש לפי שם
          </label>
          <input id="find" className="field" placeholder="חיפוש לפי שם" value={query} onChange={(e) => setQuery(e.target.value)} />
          {byGroup.length === 0 && <p className="quiet">אין חניך בשם הזה.</p>}
          {byGroup.map(([groupId, trainees]) => (
            <div key={groupId}>
              <h3>{academy.groups.get(groupId)?.name ?? `קבוצה ${groupId}`}</h3>
              {trainees.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  aria-pressed={selected === t.id}
                  className={t.status === "active" ? undefined : "left"}
                  onClick={() => setSelected(t.id)}
                >
                  {t.first_name} {t.last_name}
                  {STATUS[t.status] && ` (${STATUS[t.status]})`}
                </button>
              ))}
            </div>
          ))}
        </div>
        {selected === null ? (
          <p className="quiet">בחרו חניך כדי לראות מטרות, הערות ונוכחות.</p>
        ) : (
          <Profile key={selected} trainee={academy.trainees.get(selected)!} academy={academy} />
        )}
      </div>
    </>
  );
}

function Profile({ trainee, academy }: { trainee: Trainee; academy: Academy }) {
  const [goals, setGoals] = useState<Goal[] | null>(null);
  const [notes, setNotes] = useState<Note[] | null>(null);
  const [attendance, setAttendance] = useState<AttendanceSummary | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.goals(trainee.id), api.notes(trainee.id), api.traineeAttendance(trainee.id), api.traineeTasks(trainee.id)])
      .then(([g, n, a, t]) => {
        setGoals(g);
        setNotes([...n].reverse());
        setAttendance(a);
        setTasks(t.filter((task) => task.status === "open"));
      })
      .catch((error: Error) => setProblem(error.message));
  }, [trainee.id]);

  if (problem) return <p className="error">{problem}</p>;
  if (!goals || !notes) return <p className="quiet">טוען...</p>;

  const missed = attendance ? Object.entries(attendance.missed_by_title).slice(0, 3) : [];
  return (
    <div className="profile">
      <h2>
        {trainee.first_name} {trainee.last_name}
      </h2>
      <p className="quiet">{academy.groups.get(trainee.group_id)?.name}</p>

      <section>
        <h3>נוכחות, 30 הימים האחרונים</h3>
        {attendance?.percent == null ? (
          <p className="quiet">עוד לא סומנה נוכחות.</p>
        ) : (
          <div className="attendance-line">
            <span className="big-number">{attendance.percent}%</span>
            <span>
              הגיע ל-{attendance.present} מתוך {attendance.present + attendance.absent}.
              {attendance.excused > 0 && ` ${attendance.excused} היעדרויות מוצדקות, לא נספרות.`}
              {missed.length > 0 && (
                <>
                  <br />
                  החסיר הכי הרבה: {missed.map(([title, n]) => `${title} (${n})`).join(", ")}
                </>
              )}
            </span>
          </div>
        )}
      </section>

      {tasks.length > 0 && (
        <section>
          <h3>משימות פתוחות</h3>
          <ul className="ruled">
            {tasks.map((t) => (
              <li key={t.id}>
                {t.description}
                {t.due_date && <span className="quiet"> ({deadlineLabel(t.due_date, today())})</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h3>מטרות</h3>
        {goals.length === 0 && <p className="quiet">עוד אין מטרות.</p>}
        <ul className="ruled">
          {goals.map((g) => (
            <li key={g.id} className="goal">
              <span>
                {g.title} <span className="quiet">({GOAL_STATUS[g.status]})</span>
              </span>
              <strong>{g.progress}%</strong>
              <div className={g.progress < 30 && g.status !== "achieved" ? "bar low" : "bar"} aria-hidden="true">
                <span style={{ width: `${g.progress}%` }} />
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>הערות מאושרות</h3>
        {notes.length === 0 && <p className="quiet">עוד אין הערות.</p>}
        <ul className="ruled">
          {notes.map((n) => (
            <li key={n.id}>
              <span className={`note-kind ${n.kind}`}>{NOTE_KIND[n.kind]}</span>
              {n.text}
              <span className="quiet"> ({new Date(n.created_at).toLocaleDateString("he-IL")})</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

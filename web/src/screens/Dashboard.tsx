import { useEffect, useState } from "react";

import { api, type Approval, type AttendanceSummary, type Task } from "../api";
import type { Academy } from "../App";

const LOW = 85; // below this, a trainee's attendance is marked for a look

export function Dashboard({ academy }: { academy: Academy }) {
  const manager = academy.me.role === "manager";
  const [rates, setRates] = useState<AttendanceSummary[] | null>(null);
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api
      .attendance(academy.cohortId)
      .then(setRates)
      .catch((error: Error) => setProblem(error.message));
    api.myTasks().then(setTasks).catch(() => setTasks([]));
    if (manager) api.pendingApprovals().then(setApprovals).catch(() => setApprovals([]));
  }, [academy.cohortId, manager]);

  // A group's rate is its sessions, not the mean of its trainees' rates: present / (present + absent).
  const groups = rates
    ? [...new Set(rates.map((r) => r.group_id))].map((id) => {
        const mine = rates.filter((r) => r.group_id === id);
        const present = mine.reduce((n, r) => n + r.present, 0);
        const counted = mine.reduce((n, r) => n + r.present + r.absent, 0);
        return { id, name: academy.groups.get(id)?.name ?? `קבוצה ${id}`, percent: counted ? Math.round((present / counted) * 100) : null };
      })
    : [];
  const open = tasks.filter((t) => t.status !== "done");

  return (
    <>
      <header>
        <h1>תמונת מצב</h1>
        <p className="quiet">נוכחות של 30 הימים האחרונים, מהנמוכה לגבוהה: מי שצריך תשומת לב נמצא למעלה.</p>
      </header>
      {problem && <p className="error">{problem}</p>}
      {groups.length > 0 && (
        <div className="groups-at-a-glance">
          {groups.map((g) => (
            <div key={g.id}>
              <div className="big-number">{g.percent === null ? "–" : `${g.percent}%`}</div>
              <div>{g.name}</div>
            </div>
          ))}
        </div>
      )}
      <div className="columns">
        <section>
          <h2>נוכחות לפי חניך</h2>
          {rates === null && !problem && <p className="quiet">טוען...</p>}
          <ul className="ruled rates">
            {rates?.map((r) => (
              <li key={r.trainee_id}>
                <span>
                  {r.name}
                  {Object.keys(r.missed_by_title).length > 0 && (r.percent ?? 100) < LOW && (
                    <span className="quiet"> ({Object.keys(r.missed_by_title)[0]})</span>
                  )}
                </span>
                <div className={(r.percent ?? 100) < LOW ? "bar low" : "bar"} aria-hidden="true">
                  <span style={{ width: `${r.percent ?? 0}%` }} />
                </div>
                <strong>{r.percent === null ? "–" : `${r.percent}%`}</strong>
              </li>
            ))}
          </ul>
        </section>
        <section>
          {manager && (
            <>
              <h2>מחכים לאישור שלך</h2>
              {approvals.length === 0 ? (
                <p className="quiet">אין בקשות פתוחות.</p>
              ) : (
                <ul className="ruled">
                  {approvals.map((a) => {
                    const t = academy.trainees.get(a.trainee_id);
                    return (
                      <li key={a.id}>
                        <strong>{t ? `${t.first_name} ${t.last_name}` : `חניך ${a.trainee_id}`}</strong>
                        <br />
                        {a.reason}
                      </li>
                    );
                  })}
                </ul>
              )}
            </>
          )}
          <h2 style={{ marginTop: manager ? "2rem" : 0 }}>המשימות שלך</h2>
          {open.length === 0 ? (
            <p className="quiet">אין משימות פתוחות.</p>
          ) : (
            <ul className="ruled">
              {open.map((t) => (
                <li key={t.id}>
                  {t.description}
                  {t.due_date && <span className="quiet"> (עד {new Date(t.due_date).toLocaleDateString("he-IL")})</span>}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </>
  );
}

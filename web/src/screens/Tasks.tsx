import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";

import { api, type Staff, type Task } from "../api";
import type { Academy } from "../App";
import { daysBetween, deadlineLabel, today } from "../dates";

// The groups a task falls into, in the order they are shown. The headings say when, so they carry
// information: what is late, what is close, what can wait.
const SECTIONS = [
  { id: "late", title: "באיחור" },
  { id: "week", title: "בשבוע הקרוב" },
  { id: "later", title: "בהמשך" },
  { id: "open", title: "בלי תאריך" },
  { id: "done", title: "בוצעו" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

function sectionOf(task: Task, now: string): SectionId {
  if (task.status === "done") return "done";
  if (task.due_date === null) return "open";
  const days = daysBetween(now, task.due_date);
  if (days < 0) return "late";
  return days <= 7 ? "week" : "later";
}

export function Tasks({ academy }: { academy: Academy }) {
  const manager = academy.me.role === "manager";
  const [everyone, setEveryone] = useState(false);
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [staff, setStaff] = useState<Map<number, Staff>>(new Map());
  const [problem, setProblem] = useState<string | null>(null);
  const now = today();

  const load = useCallback(() => {
    (everyone ? api.allTasks() : api.myTasks())
      .then((list) => {
        setTasks(list);
        setProblem(null);
      })
      .catch((error: Error) => setProblem(error.message));
  }, [everyone]);

  useEffect(load, [load]);
  useEffect(() => {
    api
      .staff()
      .then((list) => setStaff(new Map(list.map((s) => [s.id, s]))))
      .catch(() => setStaff(new Map()));
  }, []);

  const sections = useMemo(() => {
    const groups = new Map<SectionId, Task[]>(SECTIONS.map((s) => [s.id, []]));
    for (const task of tasks ?? []) groups.get(sectionOf(task, now))!.push(task);
    groups.set("done", groups.get("done")!.slice(0, 10)); // the last ten are enough to look back on
    return groups;
  }, [tasks, now]);

  // Every change goes through the API, then the list is read again: what is shown is what was saved.
  const act = async (action: () => Promise<unknown>) => {
    try {
      await action();
      load();
    } catch (error) {
      setProblem((error as Error).message);
    }
  };

  return (
    <>
      <header>
        <h1>משימות</h1>
        <p className="quiet">
          משימות שכתבת, ומשימות שהגיעו מישיבות צוות. הקרוב ביותר למעלה.
          {manager ? " אפשר לכתוב משימה לכל איש צוות." : ""}
        </p>
      </header>

      <NewTaskForm academy={academy} staff={staff} onCreated={load} />

      {manager && (
        <p className="view-switch" role="group" aria-label="של מי המשימות">
          <button type="button" aria-pressed={!everyone} onClick={() => setEveryone(false)}>
            שלי
          </button>
          <button type="button" aria-pressed={everyone} onClick={() => setEveryone(true)}>
            של כל הצוות
          </button>
        </p>
      )}

      {problem && <p className="error">{problem}</p>}
      {tasks === null && !problem && <p className="quiet">טוען...</p>}
      {tasks?.length === 0 && <p className="quiet">אין משימות. משימה חדשה נכתבת בטופס שלמעלה.</p>}

      {SECTIONS.map(({ id, title }) => {
        const list = sections.get(id) ?? [];
        if (list.length === 0) return null;
        return (
          <section key={id} className={`task-section is-${id}`}>
            <h2>
              {title} <span className="count">{list.length}</span>
            </h2>
            <ul className="ruled tasks">
              {list.map((task) => (
                <TaskRow
                  key={task.id}
                  task={task}
                  academy={academy}
                  staff={staff}
                  showOwner={everyone}
                  now={now}
                  act={act}
                />
              ))}
            </ul>
          </section>
        );
      })}
    </>
  );
}

function NewTaskForm({ academy, staff, onCreated }: { academy: Academy; staff: Map<number, Staff>; onCreated: () => void }) {
  const manager = academy.me.role === "manager";
  const [description, setDescription] = useState("");
  const [due, setDue] = useState("");
  const [trainee, setTrainee] = useState("");
  const [owner, setOwner] = useState(String(academy.me.id));
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const trainees = useMemo(
    () =>
      [...academy.trainees.values()]
        .filter((t) => t.status === "active")
        .sort((a, b) => `${a.first_name} ${a.last_name}`.localeCompare(`${b.first_name} ${b.last_name}`, "he")),
    [academy.trainees],
  );

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      await api.createTask({
        description: description.trim(),
        due_date: due || null,
        trainee_id: trainee ? Number(trainee) : null,
        owner_id: manager ? Number(owner) : null,
      });
      setDescription("");
      setDue("");
      setTrainee("");
      onCreated();
    } catch (error) {
      setProblem((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className={manager ? "new-task manager" : "new-task"} onSubmit={submit}>
      <label className="wide">
        <span>מה צריך לעשות</span>
        <input
          className="field"
          value={description}
          maxLength={300}
          placeholder="למשל: לשבת עם בן על תפקיד אחראי הציוד"
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>
      <label>
        <span>עד מתי</span>
        <input className="field" type="date" min={today()} value={due} onChange={(e) => setDue(e.target.value)} />
      </label>
      <label>
        <span>על חניך</span>
        <select className="field" value={trainee} onChange={(e) => setTrainee(e.target.value)}>
          <option value="">בלי</option>
          {trainees.map((t) => (
            <option key={t.id} value={t.id}>
              {t.first_name} {t.last_name}
            </option>
          ))}
        </select>
      </label>
      {manager && (
        <label>
          <span>של מי</span>
          <select className="field" value={owner} onChange={(e) => setOwner(e.target.value)}>
            {[...staff.values()].map((s) => (
              <option key={s.id} value={s.id}>
                {s.id === academy.me.id ? `${s.full_name} (אני)` : s.full_name}
              </option>
            ))}
          </select>
        </label>
      )}
      <button className="button" type="submit" disabled={busy || description.trim().length < 5}>
        {busy ? "שומר..." : "הוספת משימה"}
      </button>
      {problem && <p className="error wide">{problem}</p>}
    </form>
  );
}

interface RowProps {
  task: Task;
  academy: Academy;
  staff: Map<number, Staff>;
  showOwner: boolean;
  now: string;
  act: (action: () => Promise<unknown>) => Promise<void>;
}

function TaskRow({ task, academy, staff, showOwner, now, act }: RowProps) {
  const [moving, setMoving] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const done = task.status === "done";
  const trainee = task.trainee_id === null ? null : academy.trainees.get(task.trainee_id);
  const canDelete =
    task.source_meeting_id === null && (task.created_by_id === academy.me.id || academy.me.role === "manager");

  const meta: string[] = [];
  if (task.due_date && !done) meta.push(deadlineLabel(task.due_date, now));
  if (trainee) meta.push(`על ${trainee.first_name} ${trainee.last_name}`);
  if (showOwner) meta.push(staff.get(task.owner_id)?.full_name ?? `איש צוות ${task.owner_id}`);
  if (task.source_meeting_id !== null) meta.push("מישיבת צוות");
  else if (task.created_by_id !== task.owner_id) meta.push(`נכתבה על ידי ${staff.get(task.created_by_id)?.full_name ?? "המנהל/ת"}`);

  return (
    <li className={done ? "task done" : "task"}>
      <button
        type="button"
        className="tick"
        aria-label={done ? "בוצעה" : `סימון כבוצעה: ${task.description}`}
        disabled={done}
        onClick={() => void act(() => api.completeTask(task.id))}
      />
      <div className="task-body">
        <span className="task-text">{task.description}</span>
        {meta.length > 0 && <span className="quiet task-meta">{meta.join(". ")}</span>}
        {moving && (
          <span className="task-move">
            <input
              className="field"
              type="date"
              min={now}
              defaultValue={task.due_date ?? ""}
              aria-label="תאריך חדש"
              onChange={(e) => {
                setMoving(false);
                void act(() => api.rescheduleTask(task.id, e.target.value || null));
              }}
            />
            {task.due_date && (
              <button className="link" type="button" onClick={() => void act(() => api.rescheduleTask(task.id, null))}>
                בלי תאריך
              </button>
            )}
          </span>
        )}
      </div>
      {!done && (
        <span className="task-actions">
          <button className="link" type="button" onClick={() => setMoving((m) => !m)}>
            {moving ? "ביטול" : "שינוי תאריך"}
          </button>
          {canDelete &&
            (confirming ? (
              <button className="link danger" type="button" onClick={() => void act(() => api.deleteTask(task.id))}>
                כן, למחוק
              </button>
            ) : (
              <button className="link" type="button" onClick={() => setConfirming(true)}>
                מחיקה
              </button>
            ))}
        </span>
      )}
    </li>
  );
}

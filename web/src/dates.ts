// Dates as the academy sees them: ISO days (YYYY-MM-DD) in local time, weeks from Sunday.

export function isoDay(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function today(): string {
  return isoDay(new Date());
}

export function addDays(day: string, days: number): string {
  const [y, m, d] = day.split("-").map(Number);
  return isoDay(new Date(y, m - 1, d + days));
}

export function sundayOf(day: string): string {
  const [y, m, d] = day.split("-").map(Number);
  const date = new Date(y, m - 1, d);
  return addDays(day, -date.getDay());
}

export function shortDate(day: string): string {
  const [, m, d] = day.split("-");
  return `${Number(d)}.${Number(m)}`;
}

export const DAY_NAMES = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export function daysBetween(from: string, to: string): number {
  const [a, b] = [from, to].map((day) => {
    const [y, m, d] = day.split("-").map(Number);
    return Date.UTC(y, m - 1, d);
  });
  return Math.round((b - a) / 86_400_000);
}

export function dayName(day: string): string {
  const [y, m, d] = day.split("-").map(Number);
  return DAY_NAMES[new Date(y, m - 1, d).getDay()];
}

// How a deadline reads: "היום", "מחר", "יום חמישי 29.10", "באיחור של 2 ימים".
export function deadlineLabel(due: string, todayIs: string): string {
  const days = daysBetween(todayIs, due);
  if (days < 0) return days === -1 ? "באיחור של יום" : `באיחור של ${-days} ימים`;
  if (days === 0) return "היום";
  if (days === 1) return "מחר";
  return `${dayName(due)} ${shortDate(due)}`;
}

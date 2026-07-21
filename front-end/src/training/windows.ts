/** Build a same-origin pop-out URL for one job's authenticated log viewer. */
export function trainingLogWindowUrl(currentUrl: string, jobId: string): string {
  const url = new URL(currentUrl);
  url.searchParams.set("training-log", jobId);
  return url.toString();
}

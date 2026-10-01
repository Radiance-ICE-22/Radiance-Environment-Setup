// Polled data the whole window shares (Explorer, Output, status bar, Home tab).
import { createContext, ReactNode, useContext } from "react";
import { api, CohortSummary, ConfigItem, Job, Machine, SceneSummary, svApi } from "../api";
import { usePoll } from "../components";

export interface Health { ok: boolean; current_job: number | null; env_script: boolean; pipeline: boolean }
export interface AppData {
  scenes: SceneSummary[]; courses: ConfigItem[]; cohorts: CohortSummary[]; jobs: Job[]; runs: any[];
  machine: Machine | null; health: Health | null; healthErr: string | null;
  gpuHistory: { t: number; used: number; util: number }[];
  reload: (what: "scenes" | "courses" | "cohorts" | "jobs" | "runs" | "all") => void;
}
const Ctx = createContext<AppData | null>(null);
export const useAppData = () => useContext(Ctx)!;

const hist: { t: number; used: number; util: number }[] = [];
export function AppDataProvider({ children }: { children: ReactNode }) {
  const scenes = usePoll(api.scenes, 15000);
  const courses = usePoll(() => api.configs("courses"), 30000);
  const cohorts = usePoll(svApi.cohorts, 15000);
  const jobs = usePoll(() => api.jobs(), 3000);
  const runs = usePoll(() => api.runs(), 20000);
  const machine = usePoll(api.machine, 5000);
  const health = usePoll(api.health, 10000);
  const live = machine.data?.gpu.live;
  if (live && (!hist.length || Date.now() - hist[hist.length - 1].t > 4000)) {
    hist.push({ t: Date.now(), used: live.used_mib, util: live.util_pct }); if (hist.length > 360) hist.shift();
  }
  const reload: AppData["reload"] = (w) => {
    if (w === "scenes" || w === "all") scenes.reload();
    if (w === "courses" || w === "all") courses.reload();
    if (w === "cohorts" || w === "all") cohorts.reload();
    if (w === "jobs" || w === "all") jobs.reload();
    if (w === "runs" || w === "all") runs.reload();
  };
  const v: AppData = {
    scenes: scenes.data ?? [], courses: courses.data ?? [], cohorts: cohorts.data ?? [], jobs: jobs.data ?? [], runs: runs.data ?? [],
    machine: machine.data, health: health.data, healthErr: health.err ?? machine.err ?? null, gpuHistory: hist, reload,
  };
  return <Ctx.Provider value={v}>{children}</Ctx.Provider>;
}

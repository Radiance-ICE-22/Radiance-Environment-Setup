// New capture document: phone video → metrically aligned splat (figs_pipeline.py up to a step).
// The Capture & Splat tab's Markers, Sampling and Reconstruct fields edit the same run.
import { useState } from "react";
import { api, FigsRun, Step, STEPS } from "../api";
import { NumField, Select, TrainOptions, useMachineTrainDefaults, usePoll, useSubmit } from "../components";
import { Problem, ToProblems, ToProperties, useCommands } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Prop, PropSection, StepStrip, Tile } from "../shell/Panes";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const STOPS = ["aruco", "sfm", "train", "verify", "bounds"] as const;
const num = (v: string) => (v.trim() === "" ? undefined : Number(v));

export default function NewCapture() {
  const videos = usePoll(api.videos, 15000);
  const d = useAppData();
  const [r, setR] = useState<FigsRun>({ scene: "", marker_id: 0, stop_after: "bounds" });
  const set = (p: Partial<FigsRun>) => setR((x) => ({ ...x, ...p }));
  useMachineTrainDefaults(d.machine?.defaults, setR);
  const { busy, err, submit } = useSubmit();

  const existing = d.scenes.map((s) => s.scene);
  const clash = r.scene && existing.find((e) => e !== r.scene && (e.includes(r.scene) || r.scene.includes(e)));
  const nameOk = NAME_RE.test(r.scene) && !existing.includes(r.scene) && !clash;
  const n = r.num_images ?? 600;
  const pairs = (n * (n - 1)) / 2;
  const ready = nameOk && !!r.video && !!r.marker_length;
  const why = !r.scene ? "Name the scene first." : !nameOk ? "Fix the scene name (see Problems)." : !r.video ? "Choose a video." : !r.marker_length ? "Enter the marker side length." : false;
  const video = videos.data?.find((v) => v.name === r.video);
  const stopIdx = STEPS.indexOf(r.stop_after ?? "bounds");

  useCommands({
    "run": { run: () => submit(r), disabled: busy ? "Submitting…" : why },
    "cap.probe": { run: () => submit({ ...r, only: "probe", stop_after: undefined }), disabled: !r.video ? "Choose a video." : !nameOk && "Name the scene first." },
    "cap.name": { run: () => alert(!r.scene ? "No name yet." : clash ? `“${r.scene}” and “${clash}” are substrings of each other: FiGS would match the wrong capture.` : existing.includes(r.scene) ? "That scene exists already." : NAME_RE.test(r.scene) ? "The name is free and matches no other scene." : "Letters, digits, _ and - only."), checked: false },
    "cap.marker_id": { value: r.marker_id ?? "", set: (v) => set({ marker_id: num(v) }) },
    "cap.marker_len": { value: r.marker_length ?? "", set: (v) => set({ marker_length: num(v) }) },
    "cap.marked": { value: r.num_marked ?? "", set: (v) => set({ num_marked: num(v) }) },
    "cap.images": { value: r.num_images ?? "", set: (v) => set({ num_images: num(v) }) },
    "cap.video": { value: r.video ?? "", options: (videos.data ?? []).map((v) => [v.name, `${v.name} (${v.mb} MB)`] as [string, string]), set: (v) => set({ video: v || undefined }) },
    "cap.stop": { value: r.stop_after ?? "bounds", options: [...STOPS], set: (v) => set({ stop_after: v as Step }) },
    "cap.aruco": { run: () => submit({ ...r, stop_after: "aruco" }), disabled: why },
    "splat.downscale": { value: r.downscale ?? "", set: (v) => set({ downscale: num(v) }) },
    "splat.iters": { value: r.train_iters ?? "", set: (v) => set({ train_iters: num(v) }) },
    "splat.cache": { value: r.cache_images ?? "", options: [["", "default"], "cpu", "gpu"], set: (v) => set({ cache_images: (v || undefined) as FigsRun["cache_images"] }) },
    "splat.sfm": { disabled: "Runs as part of the capture; for an existing scene, open it." },
  });

  const probs: Problem[] = [
    ...(!r.scene ? [{ severity: "info" as const, where: "scene name", message: "Name the scene (letters, digits, _ and -)." }] : []),
    ...(r.scene && !NAME_RE.test(r.scene) ? [{ severity: "error" as const, where: "scene name", message: "Letters, digits, _ and - only." }] : []),
    ...(existing.includes(r.scene) ? [{ severity: "error" as const, where: "scene name", message: "A scene with this name exists already." }] : []),
    ...(clash ? [{ severity: "error" as const, where: "scene name", message: `“${r.scene}” and “${clash}” are substrings of each other. FiGS finds captures by substring and fails on an ambiguous match.` }] : []),
    ...(!r.video ? [{ severity: videos.data?.length === 0 ? "error" as const : "info" as const, where: "video", message: videos.data?.length === 0 ? "video_captures/ is empty: copy the phone video there first." : "Choose a video." }] : []),
    ...(!r.marker_length ? [{ severity: "warning" as const, where: "marker", message: "Marker side length not set. Nothing downstream can check it: the splat scales with it." }] : []),
    ...(n > 300 ? [{ severity: "info" as const, where: "images", message: `${n} images: about ${Math.round(pairs / 44850)}× the matching time of the 300-image reference (38 min on an RTX 2080).` }] : []),
  ];

  return (
    <>
      <Tile title="Plan" icon="steps" meta={`figs_pipeline.py --scene ${r.scene || "<name>"} … --stop-after ${r.stop_after}`}>
        <StepStrip steps={STEPS.map((s, i) => ({ name: s, state: "", sub: i <= stopIdx ? "will run" : "later", range: i <= stopIdx,
          tip: i <= stopIdx ? "Runs in this capture job." : "Runs later, once a course exists (Course tab ▸ Fly)." }))}
          onPick={(s) => (STOPS as readonly string[]).includes(s) && set({ stop_after: s as Step })} />
        <p className="muted small" style={{ marginTop: 4 }}>New rooms normally stop after <span className="mono">bounds</span>: a course can only be written once the splat's frame is known.
          Click aruco, sfm, train, verify or bounds to stop there. Read <span className="mono">docs/FiGS_custom_video_guide.md</span> before filming.</p>
      </Tile>
      <div className="tiles cols-3">
        <Tile title="Capture" icon="camera">
          <div className="fields" style={{ gridTemplateColumns: "1fr" }}>
            <label className="f">Scene name
              <input value={r.scene} onChange={(e) => set({ scene: e.target.value.trim() })} placeholder="lab3" autoFocus />
              <span className={`hint ${r.scene && !nameOk ? "bad" : ""}`}>
                {!r.scene ? "letters, digits, _ and -" : !NAME_RE.test(r.scene) ? "letters, digits, _ and - only" : existing.includes(r.scene) ? "already exists" : clash ? `clashes with ${clash}` : "ok"}</span>
            </label>
            <Select label="Video (in video_captures/)" value={r.video} options={(videos.data ?? []).map((v) => v.name)} allowDefault={false}
              onChange={(v) => set({ video: v })} hint={videos.data?.length === 0 ? "folder empty: copy the phone video there first" : video ? `${video.mb} MB` : undefined} />
          </div>
        </Tile>
        <Tile title="ArUco marker" icon="marker" meta="DICT_4X4_50">
          <div className="fields">
            <NumField label="Marker ID" value={r.marker_id} min={0} max={49} onChange={(v) => set({ marker_id: v })} />
            <NumField label="Marker side (m)" value={r.marker_length} step={0.001} min={0.01} max={2} placeholder="0.18"
              onChange={(v) => set({ marker_length: v })} hint="black square only, tape-measured" />
            <NumField label="Images" value={r.num_images} min={50} max={2000} placeholder="600" onChange={(v) => set({ num_images: v })}
              hint={`${pairs.toLocaleString()} match pairs`} />
            <NumField label="Marker frames" value={r.num_marked} min={5} placeholder="40" onChange={(v) => set({ num_marked: v })} />
          </div>
          <p className="note small" style={{ marginTop: 6 }}>A wrong marker size gives a perfect splat at the wrong scale. After <span className="mono">bounds</span>, check the printed room size.</p>
        </Tile>
        <Tile title="Queue" icon="run">
          <div className="fields" style={{ gridTemplateColumns: "1fr" }}>
            <Select label="Stop after" value={r.stop_after} allowDefault={false} options={STOPS} onChange={(v) => set({ stop_after: v })} />
          </div>
          {err && <p className="err">{err}</p>}
          <div className="row" style={{ marginTop: 8 }}>
            <button className="primary" disabled={busy || !ready} onClick={() => submit(r)}><Icon name="run" size={14} style={{ verticalAlign: -2 }} /> Queue capture (F5)</button>
            <button disabled={busy || !r.video || !nameOk} onClick={() => submit({ ...r, only: "probe", stop_after: undefined })}>Probe only</button>
          </div>
        </Tile>
      </div>
      <div style={{ marginTop: 6 }}><TrainOptions r={r} set={set} vramMib={d.machine?.gpu.vram_mib} /></div>

      <ToProperties>
        <div className="props-title"><Icon name="camera" size={16} />New capture{r.scene ? `: ${r.scene}` : ""}</div>
        <PropSection title="Capture">
          <Prop k="Scene" tone={r.scene ? (nameOk ? "ok" : "bad") : "muted"}>{r.scene || "—"}</Prop>
          <Prop k="Video">{r.video ?? "—"}{video ? ` (${video.mb} MB)` : ""}</Prop>
          <Prop k="Marker">id {r.marker_id ?? 0} · {r.marker_length ? `${r.marker_length} m` : "length not set"}</Prop>
          <Prop k="Images">{n} ({pairs.toLocaleString()} pairs)</Prop>
          <Prop k="Marked frames">{r.num_marked ?? "default (40)"}</Prop>
        </PropSection>
        <PropSection title="Training">
          <Prop k="Iterations">{r.train_iters ?? "30000"}</Prop><Prop k="Downscale">{r.downscale ?? "auto"}</Prop>
          <Prop k="Image cache">{r.cache_images ?? "default"}</Prop><Prop k="Logging">{r.train_vis ?? "default"}</Prop>
          {(r.train_args ?? []).length > 0 && <Prop k="Extra" mono>{r.train_args!.join(" ")}</Prop>}
        </PropSection>
        <PropSection title="Run"><Prop k="Stop after">{r.stop_after}</Prop><Prop k="GPU">{d.machine?.gpu.name || "—"} · {d.machine?.gpu.vram_mib ?? "?"} MiB</Prop></PropSection>
      </ToProperties>
      <ToProblems items={probs} />
    </>
  );
}

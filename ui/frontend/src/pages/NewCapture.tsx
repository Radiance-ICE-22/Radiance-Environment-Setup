// New capture document: phone video → metrically aligned splat (figs_pipeline.py up to a step).
// The Capture & Splat tab's Markers, Sampling and Reconstruct fields edit the same run.
import { DragEvent, ReactNode, useEffect, useRef, useState } from "react";
import { api, ApiError, FigsRun, PartialUpload, Step, STEPS, VideoProbe } from "../api";
import { NumField, Select, TrainOptions, useMachineTrainDefaults, usePoll, useSubmit } from "../components";
import { addFiles, cancel, dismiss, eta, fmtBytes, fmtDur, isVideoFile, pause, pending, pickFiles, resume, sceneFromFile, Upload, useUploads } from "../uploads";
import { Problem, ToProblems, ToProperties, useCommands } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Dialog, Prop, PropSection, StepStrip, Tile } from "../shell/Panes";
import { cancelDrive, dismissDrive, driveActive, DriveCfg, DriveImport, driveApi, openDriveSetup, preloadGoogle, refreshDriveConfig, resumeDrive, startDriveFlow, useDrive } from "../drive";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const STOPS = ["aruco", "sfm", "train", "verify", "bounds"] as const;
const num = (v: string) => (v.trim() === "" ? undefined : Number(v));

export default function NewCapture() {
  const ups = useUploads();
  const dr = useDrive();
  const videos = usePoll(api.videos, 15000, [ups.doneCount, dr.doneCount]);
  const partials = usePoll(api.uploads, 15000, [ups.doneCount, ups.list.filter((u) => u.state === "cancelled").length]);
  const d = useAppData();
  const [r, setR] = useState<FigsRun>({ scene: "", marker_id: 0, stop_after: "bounds" });
  const set = (p: Partial<FigsRun>) => setR((x) => ({ ...x, ...p }));
  useMachineTrainDefaults(d.machine?.defaults, setR);
  const { busy, err, submit } = useSubmit();

  // a video chosen or dropped anywhere (this page, the ribbon) becomes this capture's video,
  // and names the scene if it has no name yet
  const seen = useRef(0);
  useEffect(() => {
    const la = ups.lastAdded;
    if (!la || la.at <= seen.current || Date.now() - la.at > 10000) return;   // only fresh picks/drops
    seen.current = la.at;
    setR((x) => ({ ...x, video: la.name, scene: x.scene || sceneFromFile(la.name) }));
  }, [ups.lastAdded]);

  // load Google's scripts early, so the sign-in popup opens within the click that asked for it
  useEffect(() => { if (dr.cfg?.enabled) preloadGoogle().catch(() => undefined); }, [dr.cfg?.enabled]);

  const upload: Upload | undefined = [...ups.list].reverse().find((u) => u.name === r.video && u.state !== "cancelled");
  const staged = !!videos.data?.some((v) => v.name === r.video);
  const dimp: DriveImport | undefined = [...dr.list].reverse().find((i) => i.name === r.video && i.state !== "cancelled" && !(i.state === "done" && staged));
  const uploading = (!!upload && (pending(upload) || upload.state === "paused" || upload.state === "error"))
    || (!!dimp && (driveActive(dimp) || dimp.state === "error" || dimp.state === "interrupted"));
  const probe = usePoll<VideoProbe | null>(() => (r.video && staged ? api.probeVideo(r.video) : Promise.resolve(null)), 0, [r.video, staged]);

  // "Queue after upload": hold the run until the video is on the host
  const [armed, setArmed] = useState<FigsRun | null>(null);
  useEffect(() => {
    if (armed && staged && armed.video === r.video && !uploading) { setArmed(null); submit(armed); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed, staged, uploading]);
  // a stopped transfer keeps the capture armed: Resume finishes it and the capture still queues
  useEffect(() => { if (armed && armed.video !== r.video) setArmed(null); }, [r.video, armed]);
  const stalled = upload?.state === "error" || upload?.state === "paused" || (!!dimp && (dimp.state === "error" || dimp.state === "interrupted"));

  const existing = d.scenes.map((s) => s.scene);
  const clash = r.scene && existing.find((e) => e !== r.scene && (e.includes(r.scene) || r.scene.includes(e)));
  const nameOk = NAME_RE.test(r.scene) && !existing.includes(r.scene) && !clash;
  const n = r.num_images ?? 600;
  const pairs = (n * (n - 1)) / 2;
  const formWhy = !r.scene ? "Name the scene first." : !nameOk ? "Fix the scene name (see Problems)." : !r.video ? "Choose or drop a video." : !r.marker_length ? "Enter the marker side length." : false;
  const why = formWhy || (!staged ? (uploading ? "The video is still uploading." : "That video is not on the host.") : false);
  const ready = !why;
  const video = videos.data?.find((v) => v.name === r.video);
  const stopIdx = STEPS.indexOf(r.stop_after ?? "bounds");
  const go = () => (staged ? submit(r) : uploading && !formWhy ? setArmed(r) : undefined);
  const videoNames = [...(videos.data ?? []).map((v) => v.name), ...ups.list.filter((u) => pending(u) || u.state === "paused").map((u) => u.name),
    ...dr.list.filter((i) => driveActive(i) || i.state === "interrupted" || i.state === "error").map((i) => i.name)]
    .filter((v, i, a) => a.indexOf(v) === i);
  const vNotes = probe.data ? videoNotes(probe.data, r) : [];
  const orphanCount = (partials.data?.uploads ?? []).filter((p) => !ups.list.some((u) => u.id === p.id && u.state !== "cancelled")).length;

  useCommands({
    "run": { run: go, disabled: busy ? "Submitting…" : armed ? "Queued to start when the upload finishes." : formWhy || (!staged && !uploading && "That video is not on the host.") },
    "cap.probe": { run: () => submit({ ...r, only: "probe", stop_after: undefined }), disabled: !r.video ? "Choose a video." : !staged ? "Wait for the upload to finish." : !nameOk && "Name the scene first." },
    "cap.name": { run: () => alert(!r.scene ? "No name yet." : clash ? `“${r.scene}” and “${clash}” are substrings of each other: FiGS would match the wrong capture.` : existing.includes(r.scene) ? "That scene exists already." : NAME_RE.test(r.scene) ? "The name is free and matches no other scene." : "Letters, digits, _ and - only."), checked: false },
    "cap.marker_id": { value: r.marker_id ?? "", set: (v) => set({ marker_id: num(v) }) },
    "cap.marker_len": { value: r.marker_length ?? "", set: (v) => set({ marker_length: num(v) }) },
    "cap.marked": { value: r.num_marked ?? "", set: (v) => set({ num_marked: num(v) }) },
    "cap.images": { value: r.num_images ?? "", set: (v) => set({ num_images: num(v) }) },
    "cap.video": { value: r.video ?? "", options: videoNames.map((n) => { const v = videos.data?.find((x) => x.name === n); return [n, v ? `${n} (${v.mb} MB)` : `${n} (uploading)`] as [string, string]; }), set: (v) => set({ video: v || undefined }) },
    "cap.stop": { value: r.stop_after ?? "bounds", options: [...STOPS], set: (v) => set({ stop_after: v as Step }) },
    "cap.aruco": { run: () => submit({ ...r, stop_after: "aruco" }), disabled: why },
    "cap.upload": { run: () => pickFiles() },
    "cap.drive": { run: () => { startDriveFlow(); } },
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
    ...(!r.video ? [{ severity: "info" as const, where: "video", message: videos.data?.length === 0 ? "No videos on the host yet: drop the phone video on this page, or Upload video." : "Choose a video, or drop one on this page." }] : []),
    ...(upload?.state === "error" ? [{ severity: "error" as const, where: "upload", message: upload.error ?? "upload failed" }] : []),
    ...(dimp && (dimp.state === "error" || dimp.state === "interrupted") ? [{ severity: "error" as const, where: "Google Drive", message: dimp.error ?? "import stopped" }] : []),
    ...vNotes.map((n) => ({ severity: n[0], where: "video", message: n[1] })),
    ...(!r.marker_length ? [{ severity: "warning" as const, where: "marker", message: "Marker side length not set. Nothing downstream can check it: the splat scales with it." }] : []),
    ...(n > 300 ? [{ severity: "info" as const, where: "images", message: `${n} images: about ${Math.round(pairs / 44850)}× the matching time of the 300-image reference (38 min on an RTX 2080).` }] : []),
  ];

  return (
    <DropHost>
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
            <label className="f">Video (in video_captures/ on the host)
              <span className="row" style={{ flexWrap: "nowrap" }}>
                <select style={{ flex: 1, minWidth: 0 }} value={r.video ?? ""} onChange={(e) => set({ video: e.target.value || undefined })}>
                  {!r.video && <option value="" disabled>choose…</option>}
                  {videoNames.map((n) => <option key={n} value={n}>{n}{videos.data?.some((v) => v.name === n) ? "" : " (uploading)"}</option>)}
                </select>
                <button className="small" onClick={() => pickFiles()} title="Send a video from this computer to the host">Upload…</button>
                <button className="small" onClick={() => startDriveFlow()} onMouseEnter={() => dr.cfg?.enabled && preloadGoogle().catch(() => undefined)}
                  title="Pick a video in Google Drive; the host downloads it">Drive…</button>
              </span>
              {upload && (pending(upload) || upload.state === "paused" || upload.state === "error") ? <UploadLine u={upload} wrap />
                : dimp && uploading ? <DriveLine i={dimp} cfg={dr.cfg} wrap />
                : <span className="hint">{videos.data?.length === 0 ? "none on the host yet: drop the phone video anywhere on this page" : video ? `${fmtBytes(video.bytes)}` : "or drop a video anywhere on this page"}</span>}
            </label>
            {!r.video && orphanCount > 0 && <p className="note small" style={{ margin: 0 }}>{orphanCount} upload{orphanCount > 1 ? "s" : ""} stopped part-way (Videos on the host, below). Choose the same file again to continue from where it stopped.</p>}
            {probe.data && <VideoSummary p={probe.data} />}
            {probe.err && staged && <span className="hint bad">ffprobe: {probe.err}</span>}
            {vNotes.filter((n) => n[0] !== "info").map((n, i) => <p key={i} className="note small" style={{ margin: 0 }}>{n[1]}</p>)}
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
            {!staged && uploading && !armed
              ? <button className="primary" disabled={busy || !!formWhy} onClick={() => setArmed(r)} title="The job is submitted the moment the upload completes. Keep this tab open.">
                  <Icon name="queue" size={14} style={{ verticalAlign: -2 }} /> Queue after upload (F5)</button>
              : armed
                ? <button onClick={() => setArmed(null)}><Icon name="cancel" size={14} style={{ verticalAlign: -2 }} /> Don't queue after upload</button>
                : <button className="primary" disabled={busy || !ready} onClick={() => submit(r)}><Icon name="run" size={14} style={{ verticalAlign: -2 }} /> Queue capture (F5)</button>}
            <button disabled={busy || !staged || !nameOk} onClick={() => submit({ ...r, only: "probe", stop_after: undefined })}>Probe only</button>
          </div>
          {armed && <p className="note small" style={{ marginTop: 6 }}>{stalled
            ? <>The transfer of {armed.video} has stopped. Resume it (Videos on the host) and the capture is still queued when it lands, or press Don't queue after upload.</>
            : <>Waiting for {armed.video} to reach the host; the capture is queued then. Keep this tab open.</>}</p>}
          {formWhy && <p className="muted small" style={{ marginTop: 6 }}>{formWhy}</p>}
        </Tile>
      </div>
      <VideosTile drive={dr.list} driveCfg={dr.cfg} ups={ups.list} videos={videos.data ?? []} partials={partials.data?.uploads ?? []} dir={partials.data?.dir}
        free={partials.data?.free_bytes ?? null} selected={r.video} inUse={new Set(d.jobs.filter((j) => j.status === "queued" || j.status === "running").map((j) => String(j.params.video ?? "")))}
        onUse={(v) => setR((x) => ({ ...x, video: v, scene: x.scene || sceneFromFile(v) }))}
        onChanged={() => { videos.reload(); partials.reload(); }} />
      <div style={{ marginTop: 6 }}><TrainOptions r={r} set={set} vramMib={d.machine?.gpu.vram_mib} /></div>

      <ToProperties>
        <div className="props-title"><Icon name="camera" size={16} />New capture{r.scene ? `: ${r.scene}` : ""}</div>
        <PropSection title="Capture">
          <Prop k="Scene" tone={r.scene ? (nameOk ? "ok" : "bad") : "muted"}>{r.scene || "—"}</Prop>
          <Prop k="Video" tone={r.video && !staged ? "warn" : undefined}>{r.video ?? "—"}{video ? ` (${video.mb} MB)` : upload && uploading ? ` (uploading ${pct(upload)} %)` : dimp ? ` (from Drive ${dpct(dimp)} %)` : ""}</Prop>
          {probe.data && <Prop k="Source">{describe(probe.data)}</Prop>}
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
      {dr.setupOpen && <DriveSetup cfg={dr.cfg} />}
    </DropHost>
  );
}

// ── video staging helpers ────────────────────────────────────────────────────
const pct = (u: Upload) => (u.size ? Math.floor((100 * u.sent) / u.size) : 0);

function describe(p: VideoProbe) {
  return `${p.width}×${p.height} ${(p.codec ?? "?").toUpperCase()} ${p.bit_depth}-bit${p.hdr ? " HDR" : ""} · ${p.fps ?? "?"} fps · ${p.duration ? fmtDur(p.duration) : "?"}`;
}

/** What the source means for the pipeline's transcode and sampling (figs_pipeline.py probe/transcode). */
function videoNotes(p: VideoProbe, r: FigsRun): ["info" | "warning" | "error", string][] {
  const out: ["info" | "warning" | "error", string][] = [];
  const W = 1920, H = 1080;                          // figs_pipeline.py --width/--height defaults
  if (p.width && p.height && p.height > p.width)
    out.push(["error", `Portrait video (${p.width}×${p.height} as displayed): the transcode scales every frame to ${W}×${H}, which squashes it. Film in landscape.`]);
  if (p.hdr || p.bit_depth > 8)
    out.push(["warning", `${p.bit_depth}-bit${p.hdr ? ` HDR (${p.color_transfer === "arib-std-b67" ? "HLG" : "PQ"})` : ""}: the transcode converts to 8-bit yuv420p without tone-mapping, so colours come out flatter. SfM and training normally still work; switch HDR video off on the phone for the next capture.`]);
  if (p.vfr) out.push(["info", `Variable frame rate (${p.avg_fps} fps average, ${p.fps} nominal): the transcode forces a constant 30 fps, which the ArUco sampling needs.`]);
  if (p.fps && p.fps > 31) out.push(["info", `${p.fps} fps: the transcode keeps 30 fps.`]);
  if ((p.width ?? 0) * (p.height ?? 0) > W * H * 2) out.push(["info", `${p.width}×${p.height} ${p.codec?.toUpperCase()}: the transcode to ${W}×${H} H.264 decodes on the CPU and takes several minutes.`]);
  if (p.duration) {
    const n = r.num_images ?? 600;
    if (p.duration < 45) out.push(["warning", `Only ${Math.round(p.duration)} s of video: short walk-throughs often leave parts of the room unregistered.`]);
    out.push(["info", `${n} images from ${fmtDur(p.duration)}: one every ${(p.duration / n).toFixed(2)} s.`]);
  }
  return out;
}

function VideoSummary({ p }: { p: VideoProbe }) {
  return (
    <span className="hint">{describe(p)}{p.frames ? ` · ${p.frames.toLocaleString()} frames` : ""}{p.rotation ? ` · rotated ${p.rotation}°` : ""}{p.device ? ` · ${p.device}` : ""}</span>
  );
}

function UploadLine({ u, wrap }: { u: Upload; wrap?: boolean }) {
  const st = u.state;
  return (
    <span className={`hint upline ${wrap ? "wrap" : ""}`}>
      <span className={`upbar ${st === "error" ? "bad" : st === "paused" ? "paused" : ""}`}><span style={{ width: `${pct(u)}%` }} /></span>
      <span>{st === "error" ? <span className="bad">{u.error}</span>
        : st === "paused" ? `paused at ${pct(u)} % (${fmtBytes(u.sent)} of ${fmtBytes(u.size)})`
        : st === "waiting" ? "waiting for the upload before it"
        : st === "starting" ? (u.note ?? "connecting…")
        : st === "finishing" ? "moving into video_captures/…"
        : `${pct(u)} % · ${fmtBytes(u.sent)} of ${fmtBytes(u.size)}${u.rate ? ` · ${fmtBytes(u.rate)}/s · ${fmtDur(eta(u))} left` : ""}${u.retries ? ` · ${u.note}` : u.resumedFrom ? ` · resumed at ${fmtBytes(u.resumedFrom)}` : ""}`}</span>
      {(st === "uploading" || st === "starting" || st === "waiting") && <button className="lnk" onClick={() => pause(u.key)}>Pause</button>}
      {(st === "paused" || st === "error") && !u.exists && <button className="lnk" onClick={() => resume(u.key)}>Resume</button>}
      {u.exists && <button className="lnk" onClick={() => resume(u.key, true)}>Replace</button>}
    </span>
  );
}

/** The New capture page accepts dropped videos anywhere on it. */
function DropHost({ children }: { children: ReactNode }) {
  const [over, setOver] = useState(0);
  const has = (e: DragEvent) => Array.from(e.dataTransfer.types).includes("Files");
  return (
    <div className="drop-host"
      onDragEnter={(e) => { if (has(e)) { e.preventDefault(); setOver((n) => n + 1); } }}
      onDragOver={(e) => { if (has(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } }}
      onDragLeave={(e) => { if (has(e)) setOver((n) => Math.max(0, n - 1)); }}
      onDrop={(e) => {
        if (!has(e)) return;
        e.preventDefault(); e.stopPropagation(); setOver(0);
        const files = Array.from(e.dataTransfer.files);
        const vids = files.filter(isVideoFile);
        if (!vids.length) { alert(`${files.map((f) => f.name).join(", ")}: not a video. Galley takes .mov, .mp4, .m4v, .mkv, .avi, .webm, .mts.`); return; }
        addFiles(vids);
      }}>
      {children}
      {over > 0 && <div className="drop-overlay"><div><Icon name="upload" size={32} /><b>Drop to upload to the host</b><span className="muted">into video_captures/; resumable if the connection drops</span></div></div>}
    </div>
  );
}

function VideosTile({ drive, driveCfg, ups, videos, partials, dir, free, selected, inUse, onUse, onChanged }: {
  drive: DriveImport[]; driveCfg: DriveCfg | null; ups: Upload[]; videos: { name: string; mb: number; bytes: number; modified: number }[]; partials: PartialUpload[];
  dir?: string; free: number | null; selected?: string; inUse: Set<string>; onUse: (v: string) => void; onChanged: () => void;
}) {
  const local = ups.filter((u) => u.state !== "cancelled" && !(u.state === "done" && videos.some((v) => v.name === u.name)));
  const justDone = new Map(ups.filter((u) => u.state === "done").map((u) => [u.name, u]));
  const dl = drive.filter((i) => i.state !== "cancelled" && !(i.state === "done" && videos.some((v) => v.name === i.name)));
  const fromDrive = new Map(drive.filter((i) => i.state === "done").map((i) => [i.name, i]));
  const act = (p: Promise<unknown>) => p.catch((e) => alert(e instanceof Error ? e.message : String(e)));
  const localIds = new Set(local.map((u) => u.id).filter(Boolean));
  const orphans = partials.filter((p) => !localIds.has(p.id));
  const del = async (name: string) => {
    if (!confirm(`Delete ${name} from video_captures/ on the host? Scenes already transcoded keep their own copy (gsplats/capture/).`)) return;
    try { await api.deleteVideo(name); onChanged(); } catch (e) { alert(e instanceof ApiError ? e.message : String(e)); }
  };
  const discard = async (id: string) => {
    try { await api.abortUpload(id); onChanged(); } catch (e) { alert(e instanceof ApiError ? e.message : String(e)); }
  };
  return (
    <Tile title="Videos on the host" icon="video" meta={`${dir ?? "video_captures/"}${free != null ? ` · ${fmtBytes(free)} free` : ""}`}
      actions={<span className="row" style={{ flexWrap: "nowrap", gap: 4 }}>
        <button onClick={() => startDriveFlow()} onMouseEnter={() => driveCfg?.enabled && preloadGoogle().catch(() => undefined)}
          title={driveCfg?.enabled ? "Pick videos in Google Drive; the host downloads them" : "Set up Google Drive access (one time)"}>
          <Icon name="cloud" size={12} style={{ verticalAlign: -2 }} /> Google Drive…</button>
        {driveCfg?.enabled && <button onClick={() => openDriveSetup()} title="Google Cloud client ID, API key and project number">⚙</button>}
        <button onClick={() => pickFiles()}><Icon name="upload" size={12} style={{ verticalAlign: -2 }} /> Upload video…</button></span>}>
      <div className="dropzone" onClick={() => pickFiles()}>
        <Icon name="upload" size={20} /> <span><b>Drop phone videos here</b> or click to choose. They are sent to the host in 8 MB chunks; if the connection drops or the tab closes, choose the same file again to continue where it stopped.</span>
      </div>
      <table className="grid" style={{ marginTop: 6 }}>
        <thead><tr><th>Video</th><th className="num">Size</th><th style={{ width: "45%" }}>Status</th><th /></tr></thead>
        <tbody>
          {local.map((u) => (
            <tr key={`u${u.key}`} className={u.name === selected ? "sel" : ""}>
              <td className="mono">{u.name}{u.note && u.state === "done" ? <span className="muted"> · {u.note}</span> : ""}</td>
              <td className="num">{fmtBytes(u.size)}</td>
              <td>{u.state === "done" ? <span className="ok">uploaded{u.finished && u.resumedFrom < u.size ? ` in ${fmtDur((u.finished - u.started) / 1000)}` : ""}</span> : <UploadLine u={u} />}</td>
              <td className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                {u.state === "done" && u.name !== selected && <button className="small" onClick={() => onUse(u.name)}>Use</button>}
                {pending(u) || u.state === "paused" || u.state === "error"
                  ? <button className="small danger" onClick={() => cancel(u.key)}>Cancel</button>
                  : <button className="small" onClick={() => dismiss(u.key)} title="Remove from this list">Clear</button>}
              </td>
            </tr>
          ))}
          {dl.map((i) => (
            <tr key={`g${i.id}`} className={i.name === selected ? "sel" : ""}>
              <td className="mono" title={i.drive_name ? `Google Drive: ${i.drive_name}` : undefined}><Icon name="cloud" size={12} style={{ verticalAlign: -2 }} /> {i.name}</td>
              <td className="num">{fmtBytes(i.size)}</td>
              <td><DriveLine i={i} cfg={driveCfg} /></td>
              <td className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                {i.name !== selected && driveActive(i) && <button className="small" onClick={() => onUse(i.name)}>Use</button>}
                {driveActive(i) || i.state === "error" || i.state === "interrupted"
                  ? <button className="small danger" onClick={() => act(i.state === "error" || i.state === "interrupted" ? dismissDrive(i.id) : cancelDrive(i.id))}>{driveActive(i) ? "Cancel" : "Discard"}</button>
                  : <button className="small" onClick={() => act(dismissDrive(i.id))}>Clear</button>}
              </td>
            </tr>
          ))}
          {orphans.map((p) => (
            <tr key={`p${p.id}`}>
              <td className="mono">{p.name}</td>
              <td className="num">{fmtBytes(p.size)}</td>
              <td><span className="upline"><span className="upbar paused"><span style={{ width: `${Math.floor((100 * p.offset) / p.size)}%` }} /></span>
                <span className="muted">interrupted at {fmtBytes(p.offset)}: choose the same file again to continue</span></span></td>
              <td className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                <button className="small" onClick={() => pickFiles()}>Resume…</button>
                <button className="small danger" onClick={() => discard(p.id)}>Discard</button>
              </td>
            </tr>
          ))}
          {videos.filter((v) => !local.some((u) => u.name === v.name && u.state !== "error")).map((v) => (
            <tr key={`v${v.name}`} className={v.name === selected ? "sel" : ""}>
              <td className="mono">{v.name}</td>
              <td className="num">{fmtBytes(v.bytes)}</td>
              <td className="muted">{fromDrive.has(v.name) && !justDone.has(v.name) ? <span className="ok">from Google Drive{fromDrive.get(v.name)!.finished && fromDrive.get(v.name)!.started ? ` in ${fmtDur(fromDrive.get(v.name)!.finished! - fromDrive.get(v.name)!.started!)}` : ""} · </span> : ""}
                {justDone.has(v.name) ? <span className="ok">{justDone.get(v.name)!.note ?? `uploaded in ${fmtDur((justDone.get(v.name)!.finished! - justDone.get(v.name)!.started) / 1000)}`} · </span> : ""}
                {new Date(v.modified * 1000).toLocaleString()}{inUse.has(v.name) ? " · used by a queued job" : ""}</td>
              <td className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                {v.name !== selected && <button className="small" onClick={() => onUse(v.name)}>Use</button>}
                <button className="small danger" disabled={inUse.has(v.name)} onClick={() => del(v.name)}>Delete</button>
              </td>
            </tr>
          ))}
          {!local.length && !dl.length && !orphans.length && !videos.length && <tr><td colSpan={4} className="empty">No videos on the host yet.</td></tr>}
        </tbody>
      </table>
    </Tile>
  );
}

// ── Google Drive ─────────────────────────────────────────────────────────────
const dpct = (i: DriveImport) => (i.size ? Math.floor((100 * i.received) / i.size) : 0);

function DriveLine({ i, cfg, wrap }: { i: DriveImport; cfg: DriveCfg | null; wrap?: boolean }) {
  const st = i.state;
  const bad = st === "error" || st === "interrupted";
  const eta = i.rate > 0 ? (i.size - i.received) / i.rate : NaN;
  return (
    <span className={`hint upline ${wrap ? "wrap" : ""}`}>
      <span className={`upbar ${bad ? "paused" : ""}`}><span style={{ width: `${dpct(i)}%` }} /></span>
      <span>{bad ? <span className={st === "error" ? "bad" : ""}>{i.error ?? st}</span>
        : st === "queued" ? "host download queued"
        : st === "verifying" ? "checking MD5 against Google Drive…"
        : st === "done" ? <span className="ok">downloaded by the host</span>
        : `host downloading from Drive · ${dpct(i)} % · ${fmtBytes(i.received)} of ${fmtBytes(i.size)}${i.rate ? ` · ${fmtBytes(i.rate)}/s · ${fmtDur(eta)} left` : ""}${i.error ? ` · ${i.error}` : ""}`}</span>
      {bad && cfg?.enabled && <button className="lnk" onClick={() => resumeDrive(cfg, i.id).catch((e) => alert(e instanceof Error ? e.message : String(e)))}>Resume</button>}
    </span>
  );
}

/** One-time Google Cloud set-up: OAuth client ID, API key, project number (stored on the host). */
function DriveSetup({ cfg }: { cfg: DriveCfg | null }) {
  const [f, setF] = useState({ client_id: cfg?.client_id ?? "", api_key: cfg?.api_key ?? "", app_id: cfg?.app_id ?? "" });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true); setErr(null);
    try { await driveApi.setConfig({ client_id: f.client_id.trim(), api_key: f.api_key.trim(), app_id: f.app_id.trim() }); await refreshDriveConfig(); openDriveSetup(false); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const origin = location.origin;
  return (
    <Dialog title="Google Drive set-up" onClose={() => openDriveSetup(false)}
      footer={<button className="push primary" disabled={busy || !f.client_id || !f.api_key || !f.app_id} onClick={save}>Save on the host</button>}>
      <div style={{ maxWidth: 620 }}>
        <p className="small">Galley opens Google's file picker in this browser; the host then downloads the videos you pick. That needs a Google Cloud project once (free):</p>
        <ol className="small" style={{ paddingLeft: 18, margin: "4px 0 8px" }}>
          <li>console.cloud.google.com → new project (e.g. <span className="mono">galley</span>). Note its <b>project number</b> (Dashboard / project settings).</li>
          <li>APIs &amp; Services → Library: enable <b>Google Drive API</b> and <b>Google Picker API</b>.</li>
          <li>OAuth consent screen: External, Testing; add your Google account as a test user. Scope <span className="mono">…/auth/drive.file</span> (Galley sees only files you pick).</li>
          <li>Credentials → Create OAuth client ID → Web application. Authorized JavaScript origins: <span className="mono">{origin}</span>{origin.includes("localhost") ? "" : " (Google accepts plain http only for localhost: open Galley through an SSH/VS Code forward on localhost)"}. Add every localhost port you use.</li>
          <li>Credentials → Create API key; restrict it to the Google Picker API and to the website <span className="mono">{origin}/*</span>.</li>
        </ol>
        <div className="fields" style={{ gridTemplateColumns: "1fr" }}>
          <label className="f">OAuth client ID<input value={f.client_id} placeholder="1234567890-abc….apps.googleusercontent.com" onChange={(e) => setF({ ...f, client_id: e.target.value })} /></label>
          <label className="f">API key<input value={f.api_key} placeholder="AIza…" onChange={(e) => setF({ ...f, api_key: e.target.value })} /></label>
          <label className="f">Project number<input value={f.app_id} placeholder="1234567890" onChange={(e) => setF({ ...f, app_id: e.target.value })} /></label>
        </div>
        <p className="muted small" style={{ marginTop: 6 }}>Stored in <span className="mono">~/.local/share/galley/google.toml</span> on the host, not in the repo. These values are not secrets (they are visible to any page that uses them); your Google password and files never pass through Galley, only a one-hour access token for the files you pick.</p>
        {err && <p className="err">{err}</p>}
      </div>
    </Dialog>
  );
}

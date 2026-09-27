import { useState } from "react";
import { api, FigsRun } from "../api";
import { NumField, Select, TrainOptions, usePoll, useSubmit } from "../components";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export default function NewCapture() {
  const videos = usePoll(api.videos, 0);
  const scenes = usePoll(api.scenes, 0);
  const machine = usePoll(api.machine, 0);
  const [r, setR] = useState<FigsRun>({ scene: "", marker_id: 0, stop_after: "bounds", cache_images: "cpu", train_vis: "tensorboard" });
  const set = (p: Partial<FigsRun>) => setR({ ...r, ...p });
  const { busy, err, submit } = useSubmit();

  const existing = scenes.data?.map((s) => s.scene) ?? [];
  const clash = r.scene && existing.find((e) => e !== r.scene && (e.includes(r.scene) || r.scene.includes(e)));
  const nameOk = NAME_RE.test(r.scene) && !existing.includes(r.scene);
  const n = r.num_images ?? 600;
  const pairs = (n * (n - 1)) / 2;

  return (
    <>
      <h1>New capture</h1>
      <p className="muted">Phone video → metrically aligned splat. Runs <span className="mono">figs_pipeline.py</span> up to the step you choose;
        the first run for a new room normally stops at <span className="mono">bounds</span>, because a course can only be written once the splat's frame is known.
        Read <span className="mono">docs/FiGS_custom_video_guide.md</span> before filming.</p>

      <fieldset className="panel">
        <legend>Capture</legend>
        <div className="fields">
          <label className="f">Scene name
            <input value={r.scene} onChange={(e) => set({ scene: e.target.value.trim() })} placeholder="lab3" />
            <span className={`hint ${r.scene && !nameOk ? "bad" : ""}`}>
              {!r.scene ? "letters, digits, _ and -" : !NAME_RE.test(r.scene) ? "letters, digits, _ and - only" : existing.includes(r.scene) ? "already exists" : "ok"}</span>
          </label>
          <Select label="Video (in video_captures/)" value={r.video} options={(videos.data ?? []).map((v) => v.name)} allowDefault={false}
            onChange={(v) => set({ video: v })} hint={videos.data?.length === 0 ? "folder empty: copy the phone video there first" : undefined} />
        </div>
        {clash && <p className="note small">“{r.scene}” and “{clash}” are substrings of each other. FiGS finds captures by substring and will fail on an ambiguous match: pick a disjoint name.</p>}
      </fieldset>

      <fieldset className="panel">
        <legend>ArUco marker (DICT_4X4_50)</legend>
        <div className="fields">
          <NumField label="Marker ID" value={r.marker_id} min={0} max={49} onChange={(v) => set({ marker_id: v })} />
          <NumField label="Marker side (m)" value={r.marker_length} step={0.001} min={0.01} max={2} placeholder="0.18"
            onChange={(v) => set({ marker_length: v })} hint="black square only, tape-measured" />
          <NumField label="Images" value={r.num_images} min={50} max={2000} placeholder="600" onChange={(v) => set({ num_images: v })}
            hint={`${pairs.toLocaleString()} match pairs`} />
          <NumField label="Marker frames" value={r.num_marked} min={5} placeholder="40" onChange={(v) => set({ num_marked: v })} />
        </div>
        <p className="note small">Nothing downstream can check the marker size: a wrong value gives a perfect splat at the wrong scale.
          After <span className="mono">bounds</span>, check that the printed room size matches reality.
          {n > 300 && ` ${n} images means about ${Math.round(pairs / 44850)}× the SuperGlue matching time of the 300-image reference (38 min on an RTX 2080).`}</p>
      </fieldset>

      <TrainOptions r={r} set={set} vramMib={machine.data?.gpu.vram_mib} />

      <div className="panel row">
        <Select label="Stop after" value={r.stop_after} allowDefault={false}
          options={["aruco", "sfm", "train", "verify", "bounds"] as const} onChange={(v) => set({ stop_after: v })} />
        <span className="spacer" />
        {err && <span className="err">{err}</span>}
        <button className="primary" disabled={busy || !nameOk || !r.video || !r.marker_length} onClick={() => submit(r)}>Queue capture</button>
      </div>
    </>
  );
}

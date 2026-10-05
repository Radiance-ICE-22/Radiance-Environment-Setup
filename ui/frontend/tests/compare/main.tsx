// Renders one synthetic .splat with drei's <Splat> (left) and our SplatMesh (right) from the
// same camera, so tests/compare_splat.py can diff the two canvases. Cloud check only.
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { Splat } from "@react-three/drei";
import { Suspense, useEffect, useState } from "react";
import { parseSplat, SplatData } from "../../src/splat/format";
import { SplatMesh } from "../../src/splat/SplatMesh";

const cam = { up: [0, 0, -1] as [number, number, number], position: [-2.5, 2.2, -1.8] as [number, number, number], fov: 50, near: 0.02, far: 100 };
function Look() { return null; }
function App() {
  const [d, setD] = useState<SplatData | null>(null);
  useEffect(() => { fetch("/test.splat").then((r) => r.arrayBuffer()).then((b) => setD(parseSplat(b))); }, []);
  const look = (c: any) => { c.camera.lookAt(0, 0, 0); };
  return (
    <>
      <div className="c" id="left"><Canvas gl={{ preserveDrawingBuffer: true }} camera={cam} onCreated={look}><Suspense fallback={null}><Splat src="/test.splat" alphaTest={0.02} /></Suspense><Look /></Canvas></div>
      <div className="c" id="right"><Canvas gl={{ preserveDrawingBuffer: true }} camera={cam} onCreated={look}>{d && <SplatMesh data={d} onReady={() => ((window as any).__ours = true)} />}</Canvas></div>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);

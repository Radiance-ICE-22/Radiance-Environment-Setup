"""Headless Chromium through the splat editor against server.py (:8811). Cloud check, Phase 3.
   python3 run_e2e.py <workdir> <shots dir>"""
import io, json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from PIL import Image
import numpy as np

work, shots = Path(sys.argv[1]), Path(sys.argv[2]); shots.mkdir(parents=True, exist_ok=True)
B = "http://127.0.0.1:8811"
courses = work / "figs_validation/SousVide/configs/courses"
results, fails = [], []
def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True); results.append((ok, msg))
    if not ok: fails.append(msg)

def overlay_mask(pg, shape):
    """True where an HTML overlay (legend, query bar, tags) covers the 3D canvas."""
    m = np.zeros(shape[:2], bool)
    c = pg.locator(".viewport canvas").first.bounding_box()
    for sel in (".viewport .legend", ".viewport .qbar", ".viewport .vtag", ".viewport .msgbar"):
        for el in pg.locator(sel).all():
            b = el.bounding_box()
            if b:
                x0, y0 = int(b["x"] - c["x"]), int(b["y"] - c["y"])
                m[max(0, y0):max(0, y0 + int(b["height"]) + 1), max(0, x0):max(0, x0 + int(b["width"]) + 1)] = True
    return m

def heat_px(pg, png):
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return ((r > 180) & (g > 80) & (b < 120) & (r - b > 90)) & ~overlay_mask(pg, a.shape)

def heat_fraction(png):
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((r > 180) & (g > 80) & (b < 120) & (r - b > 90)).mean())   # orange/yellow heat

with sync_playwright() as p:
    br = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1440, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(m.text))
    pg.on("response", lambda r: r.status >= 400 and not r.url.endswith("/flight") and "/flight?" not in r.url and errors.append(f"{r.status} {r.url}"))
    try:
        pg.goto(B + "/#/splat/backroom")
        # ribbon: contextual Semantics tab is shown and active
        expect(pg.locator("button.rb-tab.on")).to_have_text("Semantics", timeout=20000)
        check(True, "Semantics tab active on #/splat/backroom")
        expect(pg.locator(".vtag")).to_contain_text("Gaussians", timeout=90000)
        check(True, "splat downloaded, parsed and drawn: " + pg.locator(".vtag").inner_text())
        vp = pg.locator(".viewport canvas").first
        time.sleep(1.5); rgb_png = vp.screenshot(); (shots / "1_rgb.png").write_bytes(rgb_png)
        check(heat_fraction(rgb_png) < 0.002, f"RGB view has no heat colours ({heat_fraction(rgb_png):.4f})")
        expect(pg.get_by_text("lift: ready")).to_be_visible()
        check(True, "Features tile: lift ready")

        # query
        pg.locator("form.qbar input").fill("red box"); pg.keyboard.press("Enter")
        expect(pg.locator("table.cands tbody tr")).to_have_count(1, timeout=30000)
        expect(pg.locator(".legend")).to_contain_text("recoloured in", timeout=20000)
        time.sleep(1.5); rel_png = vp.screenshot(); (shots / "2_relevancy.png").write_bytes(rel_png)
        hf = float(heat_px(pg, rel_png).mean())
        check(hf > 0.001 and hf > 10 * heat_fraction(rgb_png), f"relevancy view lights the red box ({hf:.4f} of pixels heat-coloured)")
        leg = pg.locator(".legend").inner_text()
        check(True, "legend: " + leg.replace("\n", " | "))
        check(pg.locator("table.cands tbody tr").first.get_attribute("class").find("sel") >= 0, "top candidate selected")
        pg.locator(".props-title").first.wait_for()
        check("Candidate #1" in pg.locator(".props-title").first.inner_text(), "Properties: candidate #1 details")

        # floor and view modes through the ribbon
        pg.locator('[data-cmd="sem.v.pca"]').click()
        expect(pg.locator(".legend")).to_contain_text("PCA", timeout=20000)
        time.sleep(1.0); (shots / "3_pca.png").write_bytes(vp.screenshot()); check(True, "PCA view")
        pg.locator('[data-cmd="sem.v.rel"]').click()
        pg.locator('[data-cmd="sem.only"] input').check()
        time.sleep(1.0); (shots / "4_only.png").write_bytes(vp.screenshot()); check(True, "candidates only")
        pg.locator('[data-cmd="sem.only"] input').uncheck()

        # pick: click the middle of the red box on screen → Properties shows a Gaussian and labels
        box = vp.bounding_box()
        ys, xs = np.nonzero(heat_px(pg, rel_png))
        cx, cy = (float(np.median(xs)), float(np.median(ys))) if len(xs) else (box["width"] / 2, box["height"] / 2)
        time.sleep(4)                                   # software GL: let the recolour frame finish
        for attempt in range(2):
            pg.mouse.click(box["x"] + cx, box["y"] + cy)
            try:
                expect(pg.locator(".props-title").first).to_contain_text("Gaussian #", timeout=30000); break
            except AssertionError:
                if attempt: raise
        expect(pg.locator(".props-target table.grid td").first).to_be_visible(timeout=60000)
        labels = pg.locator(".props-target table.grid tr").all_inner_texts()
        check(bool(labels) and labels[0].split("\t")[1].strip() in ("red box", "box"), f"pick on the heat → best label: {labels[0] if labels else None!r}")
        pg.keyboard.press("Escape")

        # annotations: Query all
        pg.locator('[data-cmd="sem.annquery"]').click()
        expect(pg.get_by_text("2 of 2 hit")).to_be_visible(timeout=30000)
        check(True, "Annotations ▸ Query all: 2 of 2 hit")

        # annotate: place "blue chair" by clicking the chair, save
        pg.locator("form.qbar input").fill("blue chair"); pg.keyboard.press("Enter")
        expect(pg.locator(".tile", has_text="Candidates").locator("header .meta")).to_contain_text("“blue chair”", timeout=60000)
        time.sleep(1.2); png = vp.screenshot()
        ys, xs = np.nonzero(heat_px(pg, png))
        pg.locator('[data-cmd="sem.annotate"]').click()
        pg.locator('[data-cmd="sem.annlabel"] input').fill("blue chair"); pg.locator('[data-cmd="sem.annlabel"] input').press("Enter")
        pg.mouse.click(box["x"] + float(np.median(xs)), box["y"] + float(np.median(ys)))
        expect(pg.get_by_text("placed at")).to_be_visible(timeout=60000)
        pg.locator('[data-cmd="sem.annsave"]').click()
        time.sleep(1.0)
        q = json.loads((work / "figs_validation/SousVide/gsplats/workspace/backroom/semantics/queries.json").read_text())
        chair = next(x for x in q["queries"] if x["text"] == "blue chair")
        err = np.linalg.norm(np.array(chair["position"]) - np.array([-1.2, -1.6, -0.45])) if chair["position"] else None
        check(err is not None and err < 0.6, f"annotation placed by clicking: blue chair at {chair['position']} ({err if err is None else round(float(err), 2)} m from the true centre; surface point)")
        pg.keyboard.press("Escape")

        # send to course (new)
        pg.locator("form.qbar input").fill("red box"); pg.keyboard.press("Enter")
        expect(pg.locator(".tile", has_text="Candidates").locator("header .meta")).to_contain_text("“red box”", timeout=60000)
        pg.locator('[data-cmd="sem.send"]').click()
        expect(pg.locator(".dlg")).to_be_visible()
        pg.locator(".dlg").get_by_role("button", name="Send and open").click()
        expect(pg.locator("button.rb-tab.on")).to_have_text("Course", timeout=20000)
        check(pg.url.endswith("#/course/backroom/sem_red_box"), f"opened the course editor: {pg.url.split('#')[1]}")
        c = json.loads((courses / "sem_red_box.json").read_text())
        g, kfs = c["semantic_goal"], list(c["waypoints"]["keyframes"].values())
        check(g["query"] == "red box" and g["backend"] == "lift" and g["approach"] == [kfs[-1]["fo"][a][0] for a in range(3)],
              f"sem_red_box.json: goal {g['position']} · approach {g['approach']} = final keyframe · score {g.get('score')}")
        expect(pg.get_by_text("open in splat editor").first).to_be_visible(timeout=10000)
        check(True, "course editor: goal tile shows the query with a way back")
        pg.locator('[data-cmd="view.splat"]').click()
        expect(pg.locator(".vtag:visible").first).to_contain_text("splat:", timeout=90000)
        check(True, "course editor: Show ▸ Splat draws the shared splat: " + pg.locator(".vtag:visible").first.inner_text().split(" · ")[-1])
        time.sleep(3.0); (shots / "5_course.png").write_bytes(pg.screenshot())

        # Save and fly (F5): a figs job for the course
        pg.locator('[data-cmd="fly.go"]').click()
        time.sleep(2.5)
        jobs = json.loads(pg.evaluate("fetch('/api/jobs').then(r => r.text())"))
        fj = [j for j in jobs if j["kind"] == "figs"]
        check(bool(fj) and fj[0]["params"].get("course") == "sem_red_box", f"Fly queued job #{fj[0]['id'] if fj else '?'} for course {fj[0]['params'].get('course') if fj else None}")

        # send to an existing course with the approach keyframe appended
        pg.goto(B + "/#/splat/backroom/green%20ball")
        expect(pg.locator("table.cands tbody tr").first).to_be_visible(timeout=30000)
        check("green ball" in pg.locator("form.qbar input").input_value(), "route #/splat/<scene>/<query> runs the query")
        pg.locator('[data-cmd="sem.send"]').click()
        pg.locator(".dlg select").select_option("sem_red_box")
        pg.locator(".dlg").get_by_role("button", name="Send and open").click()
        expect(pg.locator("button.rb-tab.on")).to_have_text("Course", timeout=20000)
        time.sleep(1.5)
        c2 = json.loads((courses / "sem_red_box.json").read_text())
        k2 = list(c2["waypoints"]["keyframes"].items())
        check(len(k2) == 3 and k2[-1][0] == "goal1" and c2["semantic_goal"]["query"] == "green ball" and k2[1][1]["fo"] == [[r[0]] for r in kfs[-1]["fo"]],
              f"existing course: {[k for k, _ in k2]}, goal “{c2['semantic_goal']['query']}”, old end freed")
        tbl = pg.locator(".tile", has_text="Keyframes").locator("table.kf tbody tr")
        check(tbl.count() == 3, f"course editor reloaded the saved course ({tbl.count()} keyframes)")

        # scene page tile and Explorer entry
        pg.goto(B + "/#/scene/backroom")
        expect(pg.locator(".tile", has_text="Semantics").first).to_contain_text("ready", timeout=20000)
        check(True, "scene page: Semantics tile, lift ready")
        row = pg.locator(".tree-row", has_text="backroom").first
        if not pg.get_by_text("Semantics (splat editor)").count(): row.locator(".twisty").click()
        pg.get_by_text("Semantics (splat editor)").click()
        expect(pg.locator("button.rb-tab.on")).to_have_text("Semantics", timeout=20000)
        check(pg.url.endswith("#/splat/backroom"), "Explorer: scene ▸ Semantics (splat editor) opens the editor")

        # build job from the editor
        pg.goto(B + "/#/splat/backroom")
        pg.locator('[data-cmd="sem.build"]').click()
        pg.locator(".dlg select").first.select_option("redo")
        pg.locator(".dlg").get_by_role("button", name="Queue").click()
        time.sleep(1.0)
        expect(pg.get_by_text("lift: running")).to_be_visible(timeout=10000)
        check(True, "Build features queued a semantics job (lift: running)")
        expect(pg.get_by_text("lift: ready")).to_be_visible(timeout=60000)
        jobs = json.loads(pg.evaluate("fetch('/api/jobs').then(r => r.text())"))
        sj = [j for j in jobs if j["kind"] == "semantics"]
        check(bool(sj) and sj[0]["status"] == "succeeded" and "--redo" in sj[0]["argv"], f"semantics job #{sj[0]['id']}: {sj[0]['status']} · {' '.join(sj[0]['argv'][2:])}")

        check(not errors, f"no page errors ({len(errors)}): {errors[:3]}")
        br.close()

    except Exception as e:
        (shots / 'fail.png').write_bytes(pg.screenshot())
        print('props:', pg.locator('.props-target').inner_text()[:300].replace(chr(10), ' | '))
        raise
print(f"\n{sum(ok for ok, _ in results)} of {len(results)} passed")
sys.exit(1 if fails else 0)

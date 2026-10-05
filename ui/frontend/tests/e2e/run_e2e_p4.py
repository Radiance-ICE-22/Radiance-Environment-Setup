"""Headless Chromium, Phase 4: build FMGS from the editor, Compare lift | FMGS (linked cameras, the
right view's own query and candidates), send the FMGS candidate to a course. Against server.py
(:8811) started WITHOUT --fmgs. Cloud check.   python3 run_e2e_p4.py <workdir> <shots dir>"""
import io, json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from PIL import Image, ImageChops

work, shots = Path(sys.argv[1]), Path(sys.argv[2]); shots.mkdir(parents=True, exist_ok=True)
B = "http://127.0.0.1:8811"
results, fails = [], []
def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True); results.append((ok, msg))
    if not ok: fails.append(msg)
def diff(a, b):
    return ImageChops.difference(Image.open(io.BytesIO(a)).convert("RGB"), Image.open(io.BytesIO(b)).convert("RGB")).getbbox() is not None

with sync_playwright() as p:
    br = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1440, "height": 860})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("response", lambda r: r.status >= 400 and "/flight" not in r.url and errors.append(f"{r.status} {r.url}"))
    try:
        pg.goto(B + "/#/splat/backroom")
        expect(pg.locator(".vtag:visible").first).to_contain_text("Gaussians", timeout=90000)
        check(pg.locator('[data-cmd="sem.compare"]').is_disabled(), "Compare disabled with only the lift table")
        check(pg.get_by_text("fmgs: not built").count() == 1, "Features tile: fmgs not built")

        # build FMGS from the editor
        pg.locator('[data-cmd="sem.build"]').click()
        dlg = pg.locator(".dlg")
        check(dlg.locator("select").first.input_value() == "fmgs", "Build dialog offers fmgs first when only lift exists")
        dlg.locator("label", has_text="Training steps").locator("input").fill("300")
        dlg.get_by_role("button", name="Queue").click()
        expect(pg.get_by_text("fmgs: running")).to_be_visible(timeout=15000)
        check(True, "fmgs: running while the job runs")
        expect(pg.get_by_text("fmgs: ready")).to_be_visible(timeout=60000)
        jobs = json.loads(pg.evaluate("fetch('/api/jobs').then(r => r.text())"))
        sj = [j for j in jobs if j["kind"] == "semantics"][0]
        av = sj["argv"]
        check(sj["status"] == "succeeded" and av[av.index("--backend") + 1] == "fmgs" and av[av.index("--fmgs-steps") + 1] == "300"
              and av[av.index("--fmgs-width") + 1] == "480", f"job #{sj['id']}: {' '.join(av[2:])}")
        expect(pg.locator(".tile", has_text="Features").first).to_contain_text("faithful", timeout=10000)
        check(True, "Features tile: " + [l for l in pg.locator(".tile", has_text="Features").first.inner_text().splitlines() if l.startswith("fmgs:")][-1])

        # compare
        expect(pg.locator('[data-cmd="sem.compare"]')).to_be_enabled(timeout=10000)
        pg.locator("form.qbar input").fill("red box"); pg.keyboard.press("Enter")
        expect(pg.locator(".tile", has_text="Candidates").locator("header .meta")).to_contain_text("red box", timeout=60000)
        pg.locator('[data-cmd="sem.compare"]').click()
        views = pg.locator(".cw-view.split > .viewport")
        expect(views).to_have_count(2, timeout=10000)
        right = views.nth(1)
        expect(right.locator(".vtag")).to_contain_text("“red box”", timeout=90000)
        check("fmgs" in right.locator(".vtag").inner_text(), "right view: " + right.locator(".vtag").inner_text())
        expect(right.locator(".cmp-cands tbody tr").first).to_be_visible(timeout=30000)
        check(True, f"right view lists {right.locator('.cmp-cands tbody tr').count()} FMGS candidate(s); legend: " + right.locator(".legend").inner_text().replace("\n", " | "))
        expect(right.locator(".legend")).to_contain_text("lit", timeout=60000)
        time.sleep(4)
        r0 = right.locator("canvas").first.screenshot(); (shots / "p4_compare_0.png").write_bytes(pg.screenshot())
        # move the LEFT camera with the keyboard; the right view must follow
        pg.locator(".viewport canvas").first.hover()
        pg.mouse.click(10, 10)                                     # focus the page, not an input
        pg.keyboard.down("ArrowUp"); time.sleep(1.2); pg.keyboard.up("ArrowUp")
        time.sleep(6)
        r1 = right.locator("canvas").first.screenshot(); (shots / "p4_compare_1.png").write_bytes(pg.screenshot())
        check(diff(r0, r1), "right view follows the left camera (arrow keys on the left moved it)")

        # send the FMGS candidate
        right.locator(".cmp-cands").get_by_text("send…").first.click()
        expect(pg.locator(".dlg")).to_be_visible()
        pg.locator(".dlg label", has_text="New course name").locator("input").fill("sem_fmgs_red_box")
        pg.locator(".dlg").get_by_role("button", name="Send and open").click()
        expect(pg.locator("button.rb-tab.on")).to_have_text("Course", timeout=30000)
        c = json.loads((work / "figs_validation/SousVide/configs/courses/sem_fmgs_red_box.json").read_text())
        check(c["semantic_goal"]["backend"] == "fmgs" and c["semantic_goal"]["query"] == "red box", f"course goal from the right view: backend {c['semantic_goal']['backend']}, score {c['semantic_goal'].get('score')}")

        pg.goto(B + "/#/scene/backroom")
        expect(pg.locator(".tile", has_text="Semantics").first).to_contain_text("faithful", timeout=20000)
        check(True, "scene page Semantics tile: fmgs ready, faithful")
        check(not errors, f"no page errors ({len(errors)}): {errors[:3]}")
    except Exception:
        (shots / "p4_fail.png").write_bytes(pg.screenshot())
        raise
    br.close()
print(f"\n{sum(ok for ok, _ in results)} of {len(results)} passed")
sys.exit(1 if fails else 0)

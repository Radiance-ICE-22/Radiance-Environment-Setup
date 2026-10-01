# Galley screens — Figma development plugin

Builds the five Galley screens (Course workspace, Scene & Splat, SV-Net cohort, Monitor,
Configs) into the Figma file **"Galley — Win7 Ribbon UI"**, using the components already on
its *Design System* and *Ribbon & Help* pages. It replaces the Figma MCP calls that the
Starter plan's 20-calls-a-month limit stopped.

## Run it (Figma desktop app; the browser can't load local plugins)

1. Open the file **Galley — Win7 Ribbon UI** (in your drafts).
2. Menu → **Plugins → Development → Import plugin from manifest…** and pick
   `manifest.json` from this folder.
3. Menu → **Plugins → Development → Galley screens (Win7 ribbon UI)**.
4. It takes ~10–30 s. When it closes, the **Screens** page holds five 1920 × 1080 windows.

It is safe to run again: it deletes the screens it made before and rebuilds them. It also
fixes the ribbon backgrounds (the tab strip and body only painted ~1,375 px of 1,920).

If something fails, the message at the bottom of Figma says so and a red **Errors** box
appears on the Screens page — send a screenshot of it to Claude.

`code.js` is generated; it does not need network access.

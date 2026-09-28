One TOML per host. Point the server at one with `GALLEY_MACHINE=ui/machines/<host>.toml`
(the deploy scripts pick it by hostname through `ui/deploy/host.sh`).

| File | Host | GPU | Notes |
|---|---|---|---|
| `dummy.toml` | home server | RTX 3050 Ti Laptop 4 GB | build-and-test host; `cache_images = "cpu"` |
| `intellisense08.toml` | lab workstation, over Tailscale | RTX 2080 8 GB | install at `~/Radiance/figs`; Galley bound to loopback |

Still to add: `pc5060.toml` (RTX 5060 Ti 16 GB, needs the cu128 profile). `install_figs.sh`
will generate this file in Phase 6.

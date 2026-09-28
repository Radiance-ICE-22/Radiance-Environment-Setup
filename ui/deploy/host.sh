# host.sh — where things are on this machine, for the deploy scripts. Sourced, not run.
#
#   FIGS_ROOT       PROJECT_ROOT of the FiGS install (figs_env.sh, miniconda3/, SousVide/)
#   RADIANCE_REPO   this repository's checkout
#   GALLEY_MACHINE  Galley's machine profile, ui/machines/<host>.toml
#
# Chosen by hostname; any of the three can be overridden from the environment.
case "$(hostname)" in
    intellisense08*)            # lab workstation, RTX 2080 8 GB (reached over Tailscale)
        _root=~/Radiance/figs
        _repo=~/Radiance/Radiance-Environment-Setup   # clone of Radiance-ICE-22/Radiance-Environment-Setup
        _machine=intellisense08.toml ;;
    *)                          # dummy, the home server (RTX 3050 Ti 4 GB)
        _root=~/projects/figs_validation
        _repo=~/FYP-Radiance
        _machine=dummy.toml ;;
esac
FIGS_ROOT=${FIGS_ROOT:-$_root}
RADIANCE_REPO=${RADIANCE_REPO:-$_repo}
GALLEY_MACHINE=${GALLEY_MACHINE:-$RADIANCE_REPO/ui/machines/$_machine}
export FIGS_ROOT RADIANCE_REPO GALLEY_MACHINE
unset _root _repo _machine

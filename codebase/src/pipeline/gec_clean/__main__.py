from pipeline._cli import run_stage

from .config import SECTION_MAP, GecCleanConfig
from .runner import run_gec_clean


if __name__ == "__main__":
    run_stage(
        description="Produce GEC-corrected texts for condition I + native fillers.",
        config_cls=GecCleanConfig,
        section_map=SECTION_MAP,
        runner=run_gec_clean,
        use_run_dir=False,
    )

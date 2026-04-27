from pipeline._cli import run_stage

from .config import SECTION_MAP, ErrantProfileConfig
from .runner import run_errant_profile


if __name__ == "__main__":
    run_stage(
        description="ERRANT-tag each (learner, clean) pair into a profile matrix.",
        config_cls=ErrantProfileConfig,
        section_map=SECTION_MAP,
        runner=run_errant_profile,
        use_run_dir=False,
    )

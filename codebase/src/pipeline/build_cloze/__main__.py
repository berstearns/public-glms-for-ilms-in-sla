from pipeline._cli import run_stage

from .config import SECTION_MAP, BuildClozeConfig
from .runner import run_build_cloze


if __name__ == "__main__":
    run_stage(
        description="Sample gaps and materialise ClozeItems (cond I or II).",
        config_cls=BuildClozeConfig,
        section_map=SECTION_MAP,
        runner=run_build_cloze,
        use_run_dir=False,
    )

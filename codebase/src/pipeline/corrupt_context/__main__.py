from pipeline._cli import run_stage

from .config import SECTION_MAP, CorruptContextConfig
from .runner import run_corrupt_context


if __name__ == "__main__":
    run_stage(
        description="Materialise condition III by corrupting condition I.",
        config_cls=CorruptContextConfig,
        section_map=SECTION_MAP,
        runner=run_corrupt_context,
        use_run_dir=False,
    )

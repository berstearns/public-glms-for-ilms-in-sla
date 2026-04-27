from pipeline._cli import run_stage

from .config import SECTION_MAP, ClusterErrprofConfig
from .runner import run_cluster_errprof


if __name__ == "__main__":
    run_stage(
        description="k-means over ERRANT profile vectors (ERRPROF).",
        config_cls=ClusterErrprofConfig,
        section_map=SECTION_MAP,
        runner=run_cluster_errprof,
        use_run_dir=False,
    )

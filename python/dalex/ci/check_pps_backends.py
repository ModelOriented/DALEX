"""Optional PPS integration in its compatible pandas-1.x environment."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import ppscore
import dalex as dx


class Model:
    def __init__(self, backend):
        self.backend = backend

    def predict(self, data):
        assert isinstance(
            data, {'pandas': pd.DataFrame, 'polars': pl.DataFrame, 'pyarrow': pa.Table}[self.backend]
        )
        return 2 * data["x"].to_numpy() + 0.1 * data["z"].to_numpy()


def main():
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(48, 3)), columns=["x", "z", "u"])
    matrices = []
    backends = ['pandas', 'polars', 'pyarrow']
    for backend in backends:
        data = frame if backend == "pandas" else pl.from_pandas(frame) if backend == 'polars' else pa.Table.from_pandas(frame)
        exp = dx.Explainer(Model(backend), data, model_type="regression", verbose=False)
        aspect = dx.Aspect(exp, depend_method="pps")
        native = aspect._native_depend_matrix()
        assert type(native.data) is type(data)
        matrices.append(aspect.depend_matrix)
        assert type(exp.data) is type(data)
    for matrix in matrices[1:]:
        pd.testing.assert_frame_equal(matrices[0], matrix)
    report = {
        "pandas": pd.__version__,
        "ppscore": ppscore.__version__,
        "backends": backends,
        "native_matrix_storage": True,
        "passed": True,
        "max_absolute_difference": float(
            max(np.max(np.abs(matrices[0].to_numpy() - matrix.to_numpy())) for matrix in matrices[1:])
        ),
    }
    Path("pps-backend-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

import pytest
import pandas as pd
from pathlib import Path
import numpy as np
import b17


class TestB17:
    def setup_class(self):
        self.output_path = Path("out")
        self.res_path = Path("res/01013500.txt")
        self.ref_path = Path("res/01013500_ref.txt")

    def get_src_dataframe(self):
        df = pd.read_csv(self.res_path, header=72, sep="\t").iloc[1:][
            ["agency_cd", "site_no", "peak_dt", "peak_va"]
        ]
        df["peak_dt"] = pd.to_datetime(df["peak_dt"])
        df["peak_va"] = df["peak_va"].astype(np.float32)  # * 0.02832
        return df

    def get_ref_dataframe(self):
        return pd.read_csv(
            self.ref_path,
            sep=R"\s+",
            names=[
                "Return Period",
                "Probability",
                "Dischage",
                "Upper 95%",
                "Lower 95%",
                "Expect Probability Flow",
            ],
            header=None,
            dtype=np.float64,
        )

    def test_b17_imp(self):
        df = self.get_src_dataframe()
        ref_df = self.get_ref_dataframe()
        input_data = df["peak_va"].to_numpy()

        ret = b17.b17(data_in=input_data)

        ret_df = b17.extract_ref_df(ret, is_format=True)

        data_name = [
            "Dischage",
            "Upper 95%",
            "Lower 95%",
            "Expect Probability Flow",
        ]

        diff = (ref_df[data_name] - ret_df[data_name]) / ref_df[data_name]

        mean = diff.mean(axis=0)

        assert all(mean < 1e-3)

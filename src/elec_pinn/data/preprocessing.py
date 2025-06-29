import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Preprocessor:
    """
    Loads raw electrolyzer degradation data, fits/removes
    the performance component, and returns a clean DataFrame.
    Optionally, you can visualize the fit and/or the raw data.
    """

    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: filename under ./datasets (with or without .csv/.pkl)
        """
        self.dataset_name: str = dataset_name
        self.df: pd.DataFrame = None
        self.popt: np.ndarray = None
        self.pcov: np.ndarray = None

    def load(self) -> pd.DataFrame:
        """
        Load a dataset from the "./datasets" folder as a pandas DataFrame.

        - If the name contains '.pkl', loads via pickle, renames
          ['time_h','stack_1_curr_density','stack_1_cell_voltage']
          to ['t','j','U'], and returns only those columns.
        - Otherwise treats it as CSV, appends '.csv' if needed.
        """

        name = self.dataset_name

        base_path = os.getcwd()
        os.chdir("..\..")
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        os.chdir(base_path)
        if ".pkl" in name.lower():
            path = os.path.join(datasets_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Dataset not found: {path}")
            data = pd.read_pickle(path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            rename_map = {
                "time_h": "t",
                "stack_1_curr_density": "j",
                "stack_1_cell_voltage": "U"
            }
            data = data.rename(columns=rename_map)
            data = data[["t", "j", "U"]]
        else:
            if not name.lower().endswith(".csv"):
                name += ".csv"
            path = os.path.join(datasets_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Dataset not found: {path}")
            data = pd.read_csv(path)

        self.df = data
        return self.df

    @staticmethod
    def _polarization_function(x: np.ndarray, lnj0: float, R0: float) -> np.ndarray: # , k: float) -> np.ndarray:
        """
        Simple polarization curve: U_rev + (R*T/(nF))(ln(x) - lnj0) + x*R0
        """
        U_rev = 1.23        # reversible potential [V]
        R_g = 8.314         # gas constant [J/(mol*K)]
        T = (273.15 + 60)   # temperature [K]
        n = 0.5             # charge-transfer coefficient
        F = 96485           # Faraday constant [C/mol]

        return (
            U_rev
            + 1 * (R_g * T / (n * F)) * (np.log(x) - lnj0)
            + x * R0
        )

    def fit_performance(
        self,
        t0: float,
        t1: float,
        fitting_df = False,
        plot: bool = False
    ) -> pd.DataFrame:
        """
        Fit the polarization model on data where t in (t0, t1),
        then compute and store U_perf, eta_kinetic_perf, R_ohmic_perf
        on the entire dataset.

        Args:
            t0, t1: time window for fitting
            plot:   if True, show j vs U with fit curve
        """
        if self.df is None:
            self.load()

        if isinstance(fitting_df, bool): # fitting_df provided, use that instead
            fitting_df = self.df


        # Crop for fitting
        mask = (fitting_df["t"] > t0) & (fitting_df["t"] < t1)
        cropped = fitting_df.loc[mask]
        if cropped.empty:
            raise ValueError(f"No data in t∈({t0},{t1}) to fit.")

        # Curve fit
        popt, pcov = curve_fit(
            self._polarization_function,
            cropped["j"].values,
            cropped["U"].values,
            p0=[-15.0, 3e-1] # [-15.0, 3e-1, 1]
                                )
        self.popt, self.pcov = popt, pcov

        # Compute performance terms
        j_all = self.df["j"].values
        U_perf = self._polarization_function(j_all, *popt)
        eta_kinetic_perf = U_perf - self._polarization_function(j_all, popt[0], 0.0) #, popt[-1])
        R_ohmic_perf = popt[1]

        # Attach to DataFrame
        self.df["U_perf"] = U_perf
        self.df["eta_kinetic_perf"] = eta_kinetic_perf
        self.df["R_ohmic_perf"] = R_ohmic_perf

        if plot:
            plt.figure(figsize = (5,4))
            plt.scatter(cropped["j"], cropped["U"], c="k", s=30, label="Data")
            j_lin = np.linspace(cropped["j"].min(), cropped["j"].max(), 200)
            plt.plot(
                j_lin,
                self._polarization_function(j_lin, *popt),
                "--r",
                lw=3,
                label=(
                    "Fit: j$_{\mathrm{0}}$" + f" = {np.exp(popt[0]):.2g} " + "A cm$^{-2}$, "
                    "R$_{\mathrm{0}}$" + f" = {popt[1]:.2g} Ω"
                    #f"k = {popt[-1]:.2g}"
                )
            )


            plt.xlabel("Current density (A cm$^{\mathrm{-2}}$)", fontsize = 15)
            plt.ylabel("Cell voltage (V)", fontsize = 15)
            #plt.title("Performance Model Fit")
            plt.legend(frameon = False, loc = "upper left")
            #plt.ylim([1.61, 1.745])
            #plt.grid(True, ls="--", alpha=0.5)


            plt.tight_layout()
            plt.savefig("Performance_fitting_plot.png", format="png", dpi=300, transparent=True)
            plt.show()
        return self.df

    def subtract_performance(self) -> pd.DataFrame:
        """
        Subtract the fitted performance component to yield U_deg = U - U_perf.
        """
        if self.df is None:
            self.load()

        if "U_perf" not in self.df:
            raise RuntimeError("Call fit_performance() before subtract_performance().")

        self.df["U_deg"] = self.df["U"] - self.df["U_perf"]
        return self.df

    def preprocess(
        self,
        t0: float,
        t1: float,
        fitting_df = False,
        plot_fit: bool = False,
        plot_raw: bool = False
    ) -> pd.DataFrame:
        """
        Full pipeline: load → fit performance → subtract → optional plotting.

        Args:
            t0, t1:    time window for polarization-model fitting
            fitting_df: optional df to use for fitting the data
            plot_fit:  whether to show the fit vs. data
            plot_raw:  whether to show raw U vs t and U vs j scatter plots

        Returns:
            DataFrame with columns:
            ['t','j','U','U_perf','eta_kinetic_perf','R_ohmic_perf','U_deg']
        """

        self.load()
        self.fit_performance(t0, t1, fitting_df, plot=plot_fit)

        if isinstance( fitting_df, bool ): # no reference provided
            self.subtract_performance()

        if plot_raw:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].scatter(self.df["t"], self.df["U"], s=8, c="C0")
            axes[0].set(xlabel="t", ylabel="U", title="Raw U vs. t")
            axes[0].grid(True, ls="--", alpha=0.5)

            axes[1].scatter(self.df["j"], self.df["U"], s=8, c="C1")
            axes[1].set(xlabel="j", ylabel="U", title="Raw U vs. j")
            axes[1].grid(True, ls="--", alpha=0.5)

            plt.tight_layout()
            plt.show()

        return self.df

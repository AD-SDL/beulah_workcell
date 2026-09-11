import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional

from madsci.client import WorkcellClient, DataClient
from madsci.common.types.base_types import PathLike
from madsci.experiment_application import ExperimentApplication
from madsci.experiment_application.experiment_application import ExperimentApplicationConfig
from madsci.common.types.workflow_types import WorkflowDefinition
from pydantic import Field
from rich.console import Console
import george_analysis as ga
import numpy as np
import os
import glob

MU1, MU2 = 0.2, 1.0
PRE_EDGE = dict(npre=1, pre1=-145, pre2=-50, nnorm=2,
                norm1=75.0, norm2=721.58, nvict=0)

console = Console()
def read_data():
    return [0, 0]
class FlowrateConfig(ExperimentApplicationConfig):
    workflow_directory: PathLike = (Path(__file__).parent / "workflows").resolve()
    flowrate_hours: int = 8
    data_directory: str = "/net/s9data/export/9bm/BMData/Sterbinsky/2026/September2026"
class FlowrateExperiment(ExperimentApplication):
    def anchor(self, first_scan_path):
        """Fit the self-absorption constant on the unreacted precursor.

        Run once per sample, on a scan taken before heating starts. This is
        what ties the oxidation-state scale to the known stoichiometry.
        """
        try:
            path = Path(first_scan_path)
            prefix, index = path.name.rsplit(".", 1)
            scans = ga.load_xanes(path.parent, prefix, start=int(index), stop=int(index))[0]
            self._anchor_mu_max = max(float(np.max(s.mu)) for s in scans)
            fit = ga.self_absorption_anchor_fit(
                [(s.energy, s.mu, self.anchor_valence) for s in scans],
                self.slope, self.intercept, MU1, MU2, **PRE_EDGE)
            self.selfabs_C = float(fit["C"])
            self._say(f"anchored on {path.name}: C = {self.selfabs_C:.5f}")
            return self.selfabs_C
        except Exception as error:
            self.reason = f"anchoring failed: {error}"
            self._say(f"ANCHORING FAILED: {error}")
            self._say("update() will keep returning a safe cooling ramp until "
                      "anchor() succeeds.")
            return None
    def _read_scan_oxidation_state(self, scan_path):
        """Scan file -> alpha, or (None, why not). George's procedure, unchanged."""
        try:
            scan = ga.read_ascii(str(scan_path))
            scan.energy = scan.mono_energy
            scan.mu = scan.xmap8_mnka_sum / scan.xmap8_dt_corr_i0
        except Exception as error:
            return None, f"could not read {Path(scan_path).name}: {error}"

        mu_max = float(np.max(scan.mu))
        anchor_max = getattr(self, "_anchor_mu_max", None)
        if mu_max <= 0 or (anchor_max and mu_max < 0.2 * anchor_max):
            return None, f"{Path(scan_path).name}: signal lost (beam or detector)"
        if not self.selfabs_C > mu_max:
            return None, f"{Path(scan_path).name}: saturated beyond the correction"

        try:
            scan.mu = ga.apply_self_absorption(scan.mu, self.selfabs_C, 1.0)
            ga.xafs.pre_edge(scan, **PRE_EDGE)
            edge, _, bracketed = ga.dau_edge_energy(scan, MU1, MU2)
        except Exception as error:
            return None, f"{Path(scan_path).name}: analysis failed ({error})"

        return self.intercept + self.slope * edge
    def control_desicion(self, scan_path, current_state):
        """Decide whether to continue heating or start cooling.

        Returns True if the experiment should continue heating, False if it
        should start cooling. This is based on the oxidation state of the
        sample, as determined by _read_scan_oxidation_state().
        """
        state = self._read_scan_oxidation_state(scan_path)
        if state is None:
            return current_state
        oxidizing_threshold = 2.5
        oxidizing_control = [45, 0, 0, 15]
        reducing_control = [0, 0, 60, 0]
        neutral_control = [60, 0, 0, 0]
        if state > oxidizing_threshold + 0.05:
            return reducing_control
        elif state < oxidizing_threshold - 0.05:
            return oxidizing_control
        else: 
            return neutral_control
    config = FlowrateConfig()

    def __init__(self, config: Optional[FlowrateConfig] = None):
        if config:
            self.config = config
        self.workcell_client.workcell_server_url = "http://localhost:8005"
        super().__init__()

        self.flowrate_path = self.config.workflow_directory / "set_flowrate.yaml"
        self.temp_path = self.config.workflow_directory / "set_temp.yaml"
               
    def loop(self, path, latest_controls) -> None:

        control_desicion = self.control_desicion(path, latest_controls)
        # Starts Workflow on the physical hardware
        workflow = self.workcell_client.start_workflow(
            workflow_definition=self.flowrate_path,
            json_inputs={
                "target_flowrate_1": control_desicion[0],
                "target_flowrate_2": control_desicion[1],
                "target_flowrate_3": control_desicion[2],
                "target_flowrate_4": control_desicion[3],
            },
        )
        return control_desicion
    def find_latest_file(self, directory):
        list_of_files = glob.glob(os.path.join(directory, '*')) # Get all files in the directory
        latest_file = max(list_of_files, key=os.path.getctime) # Find the latest file based on creation time
        return latest_file

        
    def run_experiment(self) -> None:
        console.print("Starting experiment...")
        latest_control = [0, 0, 60, 0]
        self.workcell_client.start_workflow(
                            workflow_definition=self.flowrate_path,
                            json_inputs={
                                "target_flowrate_1": latest_control[0],
                                "target_flowrate_2": latest_control[1],
                                "target_flowrate_3": latest_control[2],
                                "target_flowrate_4": latest_control[3],
                            },
                        )
        # self.workcell_client.start_workflow(
        #             workflow_definition=self.temp_path,
        #         )
        start_time = time.time()
        num_reads = len(os.listdir(self.config.data_directory))
        self.anchor(self.config.data_directory + "/LiO4_MnOOH_15C_He.0001")
        try:
            while time.time() - start_time < time.hours(self.config.runtime_hours): 
                while len(os.listdir(self.config.data_directory)) == num_reads:
                    time.sleep(1)
                num_reads = len(os.listdir(self.config.data_directory))
                path = self.find_latest_file(self.config.data_directory)
                latest_controls = self.loop(path, latest_controls)
                
        except Exception as e:
            self.logger.error(f"Experiment stopped: {e}")
            console.print(traceback.format_exc())

        finally:
            console.print("\nDone")

if __name__ == "__main__":
    app = FlowrateExperiment()
    app.run_experiment()
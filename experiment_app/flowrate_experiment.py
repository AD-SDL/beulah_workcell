import os
import json
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

console = Console()

class FlowrateConfig(ExperimentApplicationConfig):
    workflow_directory: PathLike = (Path(__file__).parent / "workflows").resolve()
    iterations: int = Field(default=1, description="Number of iterations to run the experiment")
class FlowrateExperiment(ExperimentApplication):
    config = FlowrateConfig()

    def __init__(self, config: Optional[FlowrateConfig] = None):
        if config:
            self.config = config
        super().__init__()

        self.yaml_path = self.config.workflow_directory / "system_control_workflow.yaml"
       
    def loop(self, iteration: int) -> None:
        
        self.logger.info(f"--- Iteration {iteration + 1} ---")
        target_flowrate = 40
        target_temperature = 600

        # Starts Workflow on the physical hardware
        workflow = self.workcell_client.start_workflow(
            workflow_definition=self.yaml_path,
            json_inputs={
                "target_flowrate": target_flowrate,
                "target_temperature": target_temperature
            },
        )

        
    def run_experiment(self) -> None:
        console.print("Starting experiment...")

        try:
            for iteration in range(self.config.iterations):
                self.loop(iteration)
                
        except Exception as e:
            self.logger.error(f"Experiment stopped: {e}")
            console.print(traceback.format_exc())

        finally:
            console.print("\nDone")

if __name__ == "__main__":
    app = FlowrateExperiment()
    app.run_experiment()
from madsci.client.node_client import NodeClient
from madsci.common.types.action_types import ActionRequest
import time

nanodac_node_client=NodeClient("http://localhost:2000")
sierra_mfc_client=NodeClient("http://localhost:2001")

set_gas_request = ActionRequest(action_name="set_gas")
set_setpoint_and_settle_request = ActionRequest(action_name="set_setpoint_and_settle")
set_temp_request = ActionRequest(action_name="set_temperature")


target_temps = []
target_sierra_setpoints = []
target_gas_values = []
delay = 10

for i in range(len(target_temps)):
	set_gas_request.args = {"index": target_gas_values[i]}
	set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoints[i]}
	set_temp_request.args = {"temperature": target_temps[i]}
	nanodac_node_client.send_action(set_temp_request)
	sierra_mfc_client.send_action(set_setpoint_and_settle_request)
	sierra_mfc_client.send_action(set_gas_request)
	time.sleep(delay)

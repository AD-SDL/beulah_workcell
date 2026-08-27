"""this loop recreates the experiment performed previously, ramping to 600 degrees every 5 seconds, the tick of the MADSci scheduler"""

from madsci.client.node_client import NodeClient
from madsci.common.types.action_types import ActionRequest
import time

nanodac_node_client=NodeClient("http://localhost:2000")
sierra_mfc_client=NodeClient("http://localhost:2001")

set_setpoint_and_settle_request = ActionRequest(action_name="set_setpoint_and_settle")
set_temp_request = ActionRequest(action_name="set_temperature")

target_time = 36000
madsci_tick = 5
start = 20
target = 600
target_temps = []
target_sierra_setpoints = 40
step = madsci_tick * (target - start) / target_time
for i in range(int(target_time/madsci_tick)):
	target_temps.append(int(start + step * i))
set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoints}
sierra_mfc_client.send_action(set_setpoint_and_settle_request)
	
for i in range(len(target_temps)):
	set_temp_request.args = {"temperature": target_temps[i]}
	nanodac_node_client.send_action(set_temp_request)
	time.sleep(1)

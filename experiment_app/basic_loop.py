"""this loop recreates the experiment performed previously, ramping to 600 degrees every 5 seconds, the tick of the MADSci scheduler"""

from madsci.client.node_client import NodeClient
from madsci.common.types.action_types import ActionRequest
import time

nanodac_node_client=NodeClient("http://localhost:2000")
sierra_mfc_1_client=NodeClient("http://localhost:2011")
sierra_mfc_2_client=NodeClient("http://localhost:2012")
sierra_mfc_3_client=NodeClient("http://localhost:2013")
sierra_mfc_4_client=NodeClient("http://localhost:2014")


set_setpoint_and_settle_request = ActionRequest(action_name="set_setpoint_and_settle")
set_temp_request = ActionRequest(action_name="set_temperature")

target_time = 36000
madsci_tick = 5
start = 20
target = 600
target_temps = []
target_sierra_setpoint_1 = 40
target_sierra_setpoint_2 = 40
target_sierra_setpoint_3 = 40
target_sierra_setpoint_4 = 40
step = madsci_tick * (target - start) / target_time
for i in range(int(target_time/madsci_tick)):
	target_temps.append(int(start + step * i))
set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoint_1}
sierra_mfc_1_client.send_action(set_setpoint_and_settle_request)

set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoint_2}
sierra_mfc_2_client.send_action(set_setpoint_and_settle_request)

set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoint_3}
sierra_mfc_3_client.send_action(set_setpoint_and_settle_request)

set_setpoint_and_settle_request.args = {"setpoint": target_sierra_setpoint_4}
sierra_mfc_4_client.send_action(set_setpoint_and_settle_request)

for i in range(len(target_temps)):
	set_temp_request.args = {"temperature": target_temps[i]}
	nanodac_node_client.send_action(set_temp_request)
	time.sleep(1)
time.sleep(8*60*60)

for i in range(len(target_temps)):
	set_temp_request.args = {"temperature": target_temps[len(target_temps)-1-i]}
	nanodac_node_client.send_action(set_temp_request)
	time.sleep(1)

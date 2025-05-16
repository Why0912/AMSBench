import requests
import json
import argparse
import re
import os
import subprocess
import time
import pandas as pd
import sys
import shutil
import signal
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


API_URL = " "

API_KEY = " "

def send_api_chat(model: str, messages: list, temperature: float = 0.5):
    headers = {
        'Accept': 'application/json',
        'Authorization': API_KEY,
        'User-Agent': '',
        'Content-Type': 'application/json'
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

# ------------------------------------------------------------------------------------
# timeout handler & argument parsing
# ------------------------------------------------------------------------------------
class TimeoutException(Exception): pass

def signal_handler(signum, frame): raise TimeoutException("timeout")
signal.signal(signal.SIGALRM, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--num_per_task', type=int, default=1)
parser.add_argument('--num_of_retry', type=int, default=3)
parser.add_argument('--num_of_done', type=int, default=0)
parser.add_argument('--task_id', type=int, default=1)
parser.add_argument('--ngspice', action='store_true', default=False)
parser.add_argument('--no_prompt', action='store_true', default=False)
parser.add_argument('--skill', action='store_true', default=False)
parser.add_argument('--no_context', action='store_true', default=False)
parser.add_argument('--no_chain', action='store_true', default=False)
parser.add_argument('--retrieval', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True, help='Specify the model name (e.g., gemini-2.5-pro-exp-03-25)')
args = parser.parse_args()


# ------------------------------------------------------------------------------------
#                         Closed-source models to test
# ------------------------------------------------------------------------------------
MODEL_LIST = [
    "model name"]

complex_task_type = ['Oscillator', 'Integrator', 'Differentiator', 'Adder', 'Subtractor', 'Schmitt']
bias_usage = """Due to the operational range of the op-amp being 0 to 5V, please connect the nodes that were originally grounded to a 2.5V DC power source.
Please increase the gain as much as possible to maintain oscillation.
"""


dc_sweep_template = """
import numpy as np
analysis = simulator.dc(V[IN_NAME]=slice(0, 5, 0.01))
fopen = open("[DC_PATH]", "w")
out_voltage = np.array(analysis.Vout)
in_voltage = np.array(analysis.V[IN_NAME])
print("out_voltage: ", out_voltage)
print("in_voltage: ", in_voltage)
for item in in_voltage:
    fopen.write(f"{item:.4f} ")
fopen.write("\\n")
for item in out_voltage:
    fopen.write(f"{item:.4f} ")
fopen.write("\\n")
fopen.close()
"""


pyspice_template = """
try:
    analysis = simulator.operating_point()
    fopen = open("[OP_PATH]", "w")
    for node in analysis.nodes.values(): 
        fopen.write(f"{str(node)}\\t{float(analysis[str(node)][0]):.6f}\\n")
    fopen.close()
except Exception as e:
    print("Analysis failed due to an error:")
    print(str(e))
"""


output_netlist_template = """
source = str(circuit)
print(source)
"""

import_template = """
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
"""


sin_voltage_source_template = """
circuit.SinusoidalVoltageSource('sin', 'Vin', circuit.gnd, 
    ac_magnitude=1@u_nV, dc_offset={0}, amplitude=1@u_nV, offset={0})
"""




# This function extracts the code from the generated content which in markdown format
def extract_code(generated_content):
    empty_code_error = 0
    assert generated_content != "", "generated_content is empty"
    regex = r".*?```.*?\n(.*?)```"
    matches = re.finditer(regex, generated_content, re.DOTALL)
    first_match = next(matches, None)
    try:
        code = first_match.group(1)
        print("code", code)
        code = "\n".join([line for line in code.split("\n") if len(line.strip()) > 0])
    except:
        code = ""
        empty_code_error = 1
        return empty_code_error, code
    # Add necessary libraries
    if not args.ngspice:
        if "from PySpice.Spice.Netlist import Circuit" not in code:
            code = "from PySpice.Spice.Netlist import Circuit\n" + code
        if "from PySpice.Unit import *" not in code:
            code = "from PySpice.Unit import *\n" + code
    new_code = ""
    for line in code.split("\n"):
        new_code += line + "\n"
        if "circuit.simulator()" in line:
            break
    
    return empty_code_error, new_code



def run_code(file):
    print("IN RUN_CODE : {}".format(file))
    simulation_error = 0
    execution_error = 0
    execution_error_info = ""
    floating_node = ""
    try:
        print("-----------------running code-----------------")
        print("file:", file)
        result = subprocess.run(["python", "-u", file], check=True, text=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        print("num of lines", len(result.stdout.split("\n")))
        print("num of error lines", len(result.stderr.split("\n")))
        if len(result.stdout.split("\n")) >= 2 and ("failed" in result.stdout.split("\n")[-2] or "failed" in result.stdout.split("\n")[-1]):
            if len(result.stdout.split("\n")) >= 2:
                if "check node" in result.stdout.split("\n")[1]:
                    simulation_error = 1
                    floating_node = result.stdout.split("\n")[1].split()[-1]
                else:
                    execution_error = 1
                    if "ERROR" in result.stdout.split("\n")[1]:
                        execution_error_info = "ERROR" + result.stdout.split("\n")[1].split("ERROR")[-1]
                    elif "Error" in result.stdout.split("\n")[1]:
                        execution_error_info = "Error" + result.stdout.split("\n")[1].split("Error")[-1]
                    if len(result.stdout.split("\n"))>=3 and "ERROR" in result.stdout.split("\n")[2]:
                        execution_error_info += "\nERROR" + result.stdout.split("\n")[2].split("ERROR")[-1]
                    elif len(result.stdout.split("\n"))>=3 and "Error" in result.stdout.split("\n")[2]:
                        execution_error_info += "\nError" + result.stdout.split("\n")[2].split("Error")[-1]
                    if len(result.stdout.split("\n"))>=4 and "ERROR" in result.stdout.split("\n")[3]:
                        execution_error_info += "\nERROR" + result.stdout.split("\n")[3].split("ERROR")[-1]
                    elif len(result.stdout.split("\n"))>=4 and "Error" in result.stdout.split("\n")[3]:
                        execution_error_info += "\nError" + result.stdout.split("\n")[3].split("Error")[-1]
            if len(result.stderr.split("\n")) >= 2:
                if "check node" in result.stderr.split("\n")[1]:
                    simulation_error = 1
                    floating_node = result.stderr.split("\n")[1].split()[-1]
                else:
                    execution_error = 1
                    if "ERROR" in result.stderr.split("\n")[1]:
                        execution_error_info = "ERROR" + result.stderr.split("\n")[1].split("ERROR")[-1]
                    elif "Error" in result.stderr.split("\n")[1]:
                        execution_error_info = "Error" + result.stderr.split("\n")[1].split("Error")[-1]
                    if len(result.stdout.split("\n"))>=3 and "ERROR" in result.stderr.split("\n")[2]:
                        execution_error_info += "\nERROR" + result.stderr.split("\n")[2].split("ERROR")[-1]
                    elif len(result.stdout.split("\n"))>=3 and "Error" in result.stdout.split("\n")[2]:
                        execution_error_info += "\nError" + result.stdout.split("\n")[2].split("Error")[-1]
                    if len(result.stdout.split("\n"))>=4 and "ERROR" in result.stderr.split("\n")[3]:
                        execution_error_info += "\nERROR" + result.stderr.split("\n")[3].split("ERROR")[-1]
                    elif len(result.stdout.split("\n"))>=4 and "Error" in result.stderr.split("\n")[3]:
                        execution_error_info += "\nError" + result.stderr.split("\n")[3].split("Error")[-1]
            if simulation_error == 1:
                execution_error = 0
            if execution_error_info == "" and execution_error == 1:
                execution_error_info = "Simulation failed."
        code_content = open(file, "r").read()
        if "circuit.X" in code_content:
            execution_error_info += "\nPlease avoid using the subcircuit (X) in the code."
        if "error" in result.stdout.lower() and not "<<NAN, error".lower() in result.stdout.lower() and simulation_error == 0:
            execution_error = 1
            execution_error_info = result.stdout + result.stderr
        return execution_error, simulation_error, execution_error_info, floating_node
    except subprocess.CalledProcessError as e:
        print(f"error when running: {e}")
        print("stderr", e.stderr, file=sys.stderr)
        if "failed" in e.stdout:
            if len(e.stderr.split("\n")) >= 2:
                if "check node" in e.stderr.split("\n")[1]:
                    simulation_error = 1
                    floating_node = e.stderr.split("\n")[1].split()[-1]
        execution_error = 1
        
        execution_error_info = e.stdout + e.stderr
        if simulation_error == 1:
            execution_error = 0
            execution_error_info = "Simulation failed."
        return execution_error, simulation_error, execution_error_info, floating_node
    except subprocess.TimeoutExpired:
        print(f"Time out error when running code.")
        execution_error = 1
        execution_error_info = "Time out error when running code.\n"
        execution_error_info = "Suggestion: Avoid letting users input in Python code.\n"
        return execution_error, simulation_error, execution_error_info, floating_node




def check_netlist(netlist_path, operating_point_path, input, output, task_id, task_type):
    warning = 0
    warning_message = ""
    # Check all the input and output nodes are in the netlist
    if not os.path.exists(operating_point_path):
        return 0, ""
    fopen_op = open(operating_point_path, 'r').read()
    for input_node in input.split(", "):
        if input_node.lower() not in fopen_op.lower():
            warning_message += "The given input node ({}) is not found in the netlist.\n".format(input_node)
            warning = 1
    for output_node in output.split(", "):
        if output_node.lower() not in fopen_op.lower():
            warning_message += "The given output node ({}) is not found in the netlist.\n".format(output_node)
            warning = 1

    if warning == 1:
        warning_message += "Suggestion: You can replace the nodes actually used for input/output with the given names. Please rewrite the corrected complete code.\n"

    if task_type == "Inverter":
        return warning, warning_message
    vdd_voltage = 5.0
    vinn_voltage = 1.0
    vinp_voltage = 1.0
    for line in fopen_op.split("\n"):
        line = line.lower()
        if line.startswith("vdd"):
            vdd_voltage = float(line.split("\t")[-1])
        if line.startswith("vinn"):
            vinn_voltage = float(line.split("\t")[-1])
        if line.startswith("vinp"):
            vinp_voltage = float(line.split("\t")[-1])
    
    if vinn_voltage != vinp_voltage:
        warning_message += "The given input voltages of Vinn and Vinp are not equal.\n"
        warning = 1
        warning_message += "Suggestion: Please make sure the input voltages are equal.\n"
    
    fopen_netlist = open(netlist_path, 'r')
    voltages = {}
    for line in fopen_op.split("\n"):
        if line.strip() == "":
            continue
        node, voltage = line.split()
        voltages[node] = float(voltage)
    voltages["0"] = 0
    voltages["gnd"] = 0

    vthn = 0.5
    vthp = 0.5
    miller_node_1 = None
    miller_node_2 = None
    resistance_exist = 0
    has_diodeload = 0
    first_stage_out = None
    for line in fopen_netlist.readlines():
        if line.startswith('.'):
            continue
        if line.startswith("C"):
            if task_id == 9:
                miller_node_1 = line.split()[1].lower()
                miller_node_2 = line.split()[2].lower()
        if line.startswith("R"):
            resistance_exist = 1
        if line.startswith("M"):
            name, drain, gate, source, bulk, model = line.split()[:6]
            name = name[1:]
            drain = drain.lower()
            source = source.lower()
            bulk = bulk.lower()
            gate = gate.lower()
            mos_type = "NMOS" if "nmos" in model.lower() else "PMOS"
            ## Common-gate
            if task_id == 4:
                if drain == "vin" or gate == "vin":
                    warning_message += (f"For a common-gate amplifier, the vin should be connected to source.\n")
                    warning_message += (f"Suggestion: Please connect the vin to the source node.\n")
                    warning = 1
            elif task_id == 3:
                if drain == "vout" or gate == "vout":
                    warning_message += (f"For a common-drain amplifier, the vout should be connected to source.\n")
                    warning_message += (f"Suggestion: Please connect the vout to the source node.\n")
                    warning = 1
            elif task_id == 10:
                if gate == drain:
                    has_diodeload = 1
                    
            elif task_id == 9:
                if gate == "vin":
                    first_stage_out = drain
            
            if mos_type == "NMOS":
                # VDS
                vds_error = 0
                if voltages[drain] == 0.0:
                    if drain.lower() == "0" or drain.lower() == "gnd":
                        warning_message += (f"Suggetions: Please avoid connect {mos_type} {name} drain to the ground.\n")
                    else:
                        vds_error = 1
                        warning_message += (f"For {mos_type} {name}, the drain node ({drain}) voltage is 0.\n")
                # VDS
                elif voltages[drain] < voltages[source]:
                    vds_error = 1
                    warning_message += (f"For {mos_type} {name}, the drain node ({drain}) voltage is lower than the source node ({source}) voltage.\n")
                if vds_error == 1:
                    warning_message += (f"Suggestion: Please set {mos_type} {name} with an activated state and make sure V_DS > V_GS - V_TH.\n")
                # VGS
                vgs_error = 0
                if voltages[gate] == voltages[source]:
                    # vgs_error = 1
                    if gate == source:
                        warning_message += (f"For {mos_type} {name}, the gate node ({gate}) is connected to the source node ({source}).\n")
                        warning_message += (f"Suggestion: Please {mos_type} {name}, please divide its gate ({gate}) and source ({source}) connection.\n")
                    else:
                        vgs_error = 1
                        warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is equal to the source node ({source}) voltage.\n")
                elif voltages[gate] < voltages[source]:
                    vgs_error = 1
                    warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is lower than the source node ({source}) voltage.\n")
                elif voltages[gate] <= voltages[source] + vthn:
                    vgs_error = 1
                    warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is lower than the source node ({source}) voltage plus the threshold voltage.\n")
                if vgs_error == 1:
                    warning_message += (f"Suggestion: Please set {mos_type} {name} with an activated state by increasing the gate voltage or decreasing the source voltage and make sure V_GS > V_TH.\n")
            if mos_type == "PMOS":
                # VDS
                vds_error = 0
                if voltages[drain] == vdd_voltage:
                    if drain.lower() == "vdd":
                        warning_message += (f"Suggestion: Please avoid connect {mos_type} {name} drain to the vdd.\n")
                    else:
                        vds_error = 1
                        warning_message += (f"For {mos_type} {name}, the drain node ({drain}) voltage is V_dd.\n")
                # VDS
                elif voltages[drain] > voltages[source]:
                    vds_error = 1
                    warning_message += (f"For {mos_type} {name}, the drain node ({drain}) voltage is higher than the source node ({source}) voltage.\n")
                if vds_error == 1:
                    warning_message += (f"Suggestion: Please set {mos_type} {name} with an activated state and make sure V_DS < V_GS - V_TH.\n")
                # VGS
                vgs_error = 0
                if voltages[gate] == voltages[source]:
                    if gate == source:
                        warning_message += (f"For {mos_type} {name}, the gate node ({gate}) is connected to the source node ({source}).\n")
                        warning_message += f"Suggestion: Please {mos_type} {name}, please divide its gate ({gate}) and source ({source}) connection.\n"
                    else:
                        vgs_error = 1
                        warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is equal to the source node ({source}) voltage.\n")
                elif voltages[gate] > voltages[source]:
                    vgs_error = 1
                    warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is higher than the source node ({source}) voltage.\n")
                elif voltages[gate] >= voltages[source] - vthp:
                    vgs_error = 1
                    warning_message += (f"For {mos_type} {name}, the gate node ({gate}) voltage is higher than the source node ({source}) voltage plus the threshold voltage.\n")
                if vgs_error == 1:
                    warning_message += (f"Suggestion: Please set {mos_type} {name} with an activated state by decreasing the gate voltage or incresing the source voltage and make sure V_GS < V_TH.\n")

    if task_id in [1, 2, 3, 4, 5, 6, 8, 13]:
        if resistance_exist == 0:
            warning_message += "There is no resistance in the netlist.\n"
            warning_message += "Suggestion: Please add a resistance load in the netlist.\n"
            warning = 1
    if task_id == 9:
        if first_stage_out == None:
            warning_message += "There is no first stage output in the netlist.\n"
            warning_message += "Suggestion: Please add a first stage output in the netlist.\n"
            warning = 1
        elif (first_stage_out == miller_node_1 and miller_node_2 == "vout") or (first_stage_out == miller_node_2 and miller_node_1 == "vout"):
            pass
        elif miller_node_1 == None:
            warning_message += "There no Miller capacitor in the netlist.\n"
            warning_message += "Suggestion: Please correctly connect the Miller compensation capacitor."
            warning = 1
        else:
            warning_message += "The Miller compensation capacitor is not correctly connected.\n"
            warning_message += "Suggestion: Please correctly connect the Miller compensation capacitor."
            warning = 1
    if task_id == 10 and has_diodeload == 0:
        warning_message += "There is no diode-connected load in the netlist.\n"
        warning_message += "Suggestion: Please add a diode-connected load in the netlist.\n"
        warning = 1
    warning_message = warning_message.strip()
    if warning_message == "":
        warning = 0
    else:
        warning = 1
        warning_message = "According to the operating point check, there are some issues, which defy the general operating principles of MOSFET devices. \n" + warning_message + "\n"
        warning_message += "\nPlease help me fix the issues and rewrite the corrected complete code.\n"
    return warning, warning_message

    
def check_function(task_id, code_path, task_type):
    fwrite_code_path = "{}_check.py".format(code_path.rsplit(".", 1)[0])
    fwrite_code = open(fwrite_code_path, 'w')
    if task_type == "CurrentMirror":
        test_code = open("problem_check/CurrentMirror.py", "r").read()
        code = open(code_path, 'r').read()
        code = code + "\n" + test_code
        fwrite_code.write(code)
        fwrite_code.close()
    elif task_type == "Amplifier" or task_type == "Opamp":
        voltage = "1.0"
        test_code = open(f"problem_check/{task_type}.py", "r").read()
        for line in open(code_path, 'r').readlines():
            if line.startswith("circuit.V") and "vin" in line.lower():
                
                parts = line.split("#")[0].strip().rstrip(")").split(",")
                raw_voltage = parts[-1].strip()
                if raw_voltage[0] == "\"" or raw_voltage[0] == "'":
                    raw_voltage = raw_voltage[1:-1]
                if "dc" in raw_voltage.lower():
                    voltage = raw_voltage.split(" ")[1]
                else:
                    voltage = raw_voltage
                new_voltage = " \"dc {} ac 1u\"".format(voltage)
                parts[-1] = new_voltage
                line = ",".join(parts) + ")\n"

            fwrite_code.write(line)
        fwrite_code.write("\n")
        fwrite_code.write(test_code)
        fwrite_code.close()
        print("voltage", voltage)
    elif task_type == "Inverter":
        test_code = open("problem_check/Inverter.py", "r").read()
        code = open(code_path, 'r').read()
        code = code + "\n" + test_code
        fwrite_code.write(code)
        fwrite_code.close()
    else:
        return 0, ""
    try:
        result = subprocess.run(["python", "-u", fwrite_code_path], check=True, text=True, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
        print("function correct.")
        func_error = 0
        return_message = ""
    except subprocess.CalledProcessError as e:
        print("function error.")
        print("e.stdout", e.stdout)
        print("e.stderr", e.stderr)
        func_error = 1
        return_message = "\n".join(e.stdout.split("\n"))

    return func_error, return_message

import numpy as np
def get_best_voltage(dc_file_path):
    fopen = open(dc_file_path, 'r')
    vin = np.array([float(x) for x in fopen.readline().strip().split(" ")])
    vout = np.array([float(x) for x in fopen.readline().strip().split(" ")])
    if np.max(vout) - np.min(vout) < 1e-3:
        return 1, 0
    min_margin = 10.0
    for i, v in enumerate(vout):
        if np.abs(v - 2.5) < min_margin:
            min_margin = np.abs(v - 2.5)
            best_voltage = vin[i]
    return 0, best_voltage


def get_vin_name(netlist_content, task_type):
    vinn_name = "in"
    vinp_name = None
    for line in netlist_content.split("\n"):
        if not line.lower().startswith("v"):
            continue
        if len(line.lower().split()) < 2:
            continue
        if task_type == "Amplifier" and "vin" in line.lower().split()[1]:
            vinn_name = line.split()[0][1:]
        if task_type == "Opamp" and "vinp" in line.lower().split()[1]:
            vinp_name = line.split()[0][1:]
        if task_type == "Opamp" and "vinn" in line.lower().split()[1]:
            vinn_name = line.split()[0][1:]
    return vinn_name, vinp_name


def replace_voltage(raw_code, best_voltage, vinn_name, vinp_name):
    new_code = ""
    for line in raw_code.split("\n"):
        if not line.lower().startswith("circuit.v"):
            new_code += line + "\n"
            continue
        if vinn_name is not None and (line.lower().startswith(f"circuit.v('{vinn_name.lower()}'") or line.lower().startswith(f"circuit.v(\"{vinn_name.lower()}\"")):
            parts = line.split("#")[0].strip().rstrip(")").split(",")
            new_voltage = " {}".format(best_voltage)
            parts[-1] = new_voltage
            line = ",".join(parts) + ")"
        elif vinp_name is not None and (line.lower().startswith(f"circuit.v('{vinp_name.lower()}'") or line.lower().startswith(f"circuit.v(\"{vinp_name.lower()}\"")):
            parts = line.split("#")[0].strip().rstrip(")").split(",")
            new_voltage = " {}".format(best_voltage)
            parts[-1] = new_voltage
            line = ",".join(parts) + ")"
        new_code += line + "\n"
    return new_code


def connect_vinn_vinp(dc_sweep_code, vinn_name, vinp_name):
    new_code = ""
    for line in dc_sweep_code.split("\n"):
        if not line.lower().startswith("circuit.v"):
            new_code += line + "\n"
            continue
        if vinp_name is not None and (line.lower().startswith(f"circuit.v('{vinp_name.lower()}'") or line.lower().startswith(f"circuit.v(\"{vinp_name.lower()}\"")):
            new_line = "circuit.V('dc', 'Vinn', 'Vinp', 0.0)\n"
            new_code += new_line
        else:
            new_code += line + "\n"
    return new_code

def get_subcircuits_info(subcircuits, 
                    lib_data_path = "lib_info.tsv", task_data_path = "problem_set.tsv"):
    lib_df = pd.read_csv(lib_data_path, delimiter='\t')
    task_df = pd.read_csv(task_data_path, delimiter='\t')
    # New data frame
    columns = ["Id", "Circuit Type", "Gain/Differential-mode gain (dB)", "Common-mode gain (dB)", "Input", "Output"]
    subcircuits_df = pd.DataFrame(columns=columns)
    # write all the subcircuits information
    for sub_id in subcircuits:
        print("sub_id", sub_id)
        lib_df_row = lib_df.loc[lib_df['Id'] == sub_id]
        task_df_row = task_df.loc[task_df['Id'] == sub_id]
        print("task_df_row", task_df_row)
        sub_type = task_df.loc[task_df['Id'] == sub_id, 'Type'].item()
        sub_gain = float(lib_df.loc[lib_df['Id'] == sub_id, 'Av (dB)'].item())
        sub_com_gan = float(lib_df.loc[lib_df['Id'] == sub_id, 'Com Av (dB)'].item())
        sub_gain = "{:.2f}".format(sub_gain)
        sub_com_gan = "{:.2f}".format(sub_com_gan)
        print("sub_gain", sub_gain)
        print("sub_com_gan", sub_com_gan)
        print("sub_id", sub_id)
        print("sub_type", sub_type)
        sub_input = task_df.loc[task_df['Id'] == sub_id, 'Input'].item()
        input_node_list = sub_input.split(", ")
        input_node_list = [node for node in input_node_list if "bias" not in node]
        sub_input = ", ".join(input_node_list)

        sub_output = task_df.loc[task_df['Id'] == sub_id, 'Output'].item()
        output_node_list = sub_output.split(", ")
        output_node_list = [node for node in output_node_list if "outn" not in node and "outp" not in node]
        sub_output = ",".join(output_node_list)
        
        new_row = {'Id': sub_id, "Circuit Type": sub_type, "Gain/Differential-mode gain (dB)": sub_gain, "Common-mode gain (dB)": sub_com_gan, "Input": sub_input, "Output": sub_output}
        subcircuits_df = pd.concat([subcircuits_df, pd.DataFrame([new_row])], ignore_index=True)
    print("subcircuits_df")
    print(subcircuits_df)
    subcircuits_info = subcircuits_df.to_csv(sep='\t', index=False)
    return subcircuits_info


def get_note_info(subcircuits,
                    lib_data_path = "lib_info.tsv", task_data_path = "problem_set.tsv"):
    lib_df = pd.read_csv(lib_data_path, delimiter='\t')
    task_df = pd.read_csv(task_data_path, delimiter='\t')
    note_info = ""

    for sub_id in subcircuits:
        sub_type = task_df.loc[task_df['Id'] == sub_id, 'Type'].item()
        sub_name = task_df.loc[task_df['Id'] == sub_id, 'Submodule Name'].item()
        sub_bias_voltage = lib_df.loc[lib_df['Id'] == sub_id, 'Voltage Bias'].item()
        if "Amplifier" not in sub_type and "Opamp" not in sub_type:
            continue
        sub_phase = lib_df.loc[lib_df['Id'] == sub_id, 'Vin(n) Phase'].item()
        if sub_type == "Amplifier":
            if sub_phase == "inverting":
                other_sub_phase = "non-inverting"
            else:
                other_sub_phase = "inverting"
            note_info += f"The Vin of {sub_name} is the {sub_phase} input.\n"
            note_info += f"There is NO in {other_sub_phase} input in {sub_name}.\n"
            note_info += f"The DC operating voltage for Vin is {sub_bias_voltage} V.\n"
        elif sub_type == "Opamp":
            if sub_phase == "inverting":
                other_sub_phase = "non-inverting"
            else:
                other_sub_phase = "inverting"
            note_info += f"The Vinn of {sub_name} is the {sub_phase} input.\n"
            note_info += f"The Vinp of {sub_name} is the {other_sub_phase} input.\n"
            note_info += f"The DC operating voltage for Vinn/Vinp is {sub_bias_voltage} V.\n"
    print("note_info", note_info)
    return note_info, sub_bias_voltage


def get_call_info(subcircuits,
                    lib_data_path = "lib_info.tsv", task_data_path = "problem_set.tsv"):
    template = '''```python
from p[ID]_lib import *
# declare the subcircuit
circuit.subcircuit([SUBMODULE_NAME]())
# create a subcircuit instance
circuit.X('1', '[SUBMODULE_NAME]', [INPUT_OUTPUT])
```
'''
    lib_df = pd.read_csv(lib_data_path, delimiter='\t')
    task_df = pd.read_csv(task_data_path, delimiter='\t')
    call_info = ""
    for it, subcircuit in enumerate(subcircuits):
        sub_id = subcircuit
        sub_name = task_df.loc[task_df['Id'] == sub_id, 'Submodule Name'].item()
        input_nodes = task_df.loc[task_df['Id'] == sub_id, 'Input'].item()
        output_nodes = task_df.loc[task_df['Id'] == sub_id, 'Output'].item()
        sub_info = template.replace('[SUBMODULE_NAME]', sub_name)
        input_node_list = input_nodes.split(", ")
        input_node_list = [node for node in input_node_list if "bias" not in node]

        # for input_node in input_nodes.split(", "):
        input_info = ", ".join([f"'{input_node}'" for input_node in input_node_list])
        output_node_list = output_nodes.split(", ")
        output_node_list = [node for node in output_node_list if "outn" not in node and "outp" not in node]
        output_info = ", ".join([f"'{output_node}'" for output_node in output_node_list])
        if input_info !=  "" and output_info != "":
            input_output = f"{input_info}, {output_info}"
        elif input_info == "":
            input_output = f"{output_info}"
        else:
            input_output = f"{input_info}"
        sub_info = sub_info.replace('[INPUT_OUTPUT]', input_output)
        sub_info = sub_info.replace('[ID]', str(sub_id))
        call_info += sub_info
    return call_info

global generator
generator = None


def write_pyspice_code(sp_code_path, code_path, op_path):
    sp_code = open(sp_code_path, 'r')
    code = open(code_path, 'w')
    code.write(import_template)
    code.write("circuit = Circuit('circuit')\n")
    for line in sp_code.readlines():
        if line.startswith(".model"):
            parts = line.split()
            if len(parts) < 6:
                continue
            code.write(f"circuit.model('{parts[1]}', '{parts[2]}', {parts[3]}, {parts[4]}, {parts[5]})\n")
        elif line.startswith('R') or line.startswith('C') or line.startswith('V') or line.startswith('I'):
            type_name = line[0]
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0][1:]
            n1 = parts[1]
            n2 = parts[2]
            value = parts[3]
            code.write(f"circuit.{type_name}('{name}', '{n1}', '{n2}', '{value}')\n")
        elif line.startswith('M'):
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0][1:]
            drain = parts[1]
            gate = parts[2]
            source = parts[3]
            bulk = parts[4]
            model = parts[5]
            w = parts[6]
            l = parts[7]
            code.write(f"circuit.MOSFET('{name}', '{drain}', '{gate}', '{source}', '{bulk}', model='{model}', {w}, {l})\n")
    code.write("simulator = circuit.simulator()\n")
    code.write(pyspice_template.replace("[OP_PATH]", op_path))
    code.close()


def start_tmux_session(session_name, command):
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])
    subprocess.run(['tmux', 'send-keys', '-t', session_name, command, 'C-m'])
    print(f"tmux session '{session_name}' started, running command: {command}")


def kill_tmux_session(session_name):
    try:
        subprocess.run(['tmux', 'kill-session', '-t', session_name], check=True)
        print(f"tmux session '{session_name}' has been killed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to kill tmux session '{session_name}'. Session might not exist.")


def work(task, inp, outp, tid, it, bg, ttype, flog, money_quota=100, subcircuits=None):
    import os
    output_root = f"outputs_pass{args.num_per_task}"
    output_dir = os.path.join(output_root, f"{args.model}_p{tid}_i{it+1}")
    os.makedirs(output_dir, exist_ok=True)

    if ttype not in complex_task_type or not args.skill:
        f = open('prompt_template.md' if not args.ngspice else 'prompt_template_ngspice.md','r')
        if args.no_prompt: f=open('prompt_template_wo_prompt.md','r')
        elif args.no_context: f=open('prompt_template_wo_context.md','r')
        elif args.no_chain: f=open('prompt_template_wo_chain_of_thought.md','r')
        prompt=f.read(); f.close()
        prompt=prompt.replace('[TASK]',task).replace('[INPUT]',inp).replace('[OUTPUT]',outp)
        if ttype in complex_task_type: prompt=prompt.replace('6. Avoid using subcircuits.','')
    else:
        f=open('prompt_template_complex.md','r'); prompt=f.read(); f.close()
        prompt=prompt.replace('[TASK]',task).replace('[INPUT]',inp).replace('[OUTPUT]',outp)
        info=get_subcircuits_info(subcircuits); note,bv=get_note_info(subcircuits)
        if ttype=='Oscillator': note+=bias_usage
        call=get_call_info(subcircuits)
        prompt=prompt.replace('[SUBCIRCUITS_INFO]',info).replace('[NOTE_INFO]',note).replace('[CALL_INFO]',call)
    if bg: prompt+=f"\nHint Background:\n{bg}\n## Answer \n"

    pe=open('execution_error.md').read(); se=open('simulation_error.md').read()

    base=[{"role":"system","content":"You are an analog integrated circuits expert."},
          {"role":"user","content":prompt}]

    answer=''; code_path=''
    for cid in range(args.num_of_retry):
        msgs=[m.copy() for m in base]
        comp = send_api_chat(args.model, msgs, args.temperature)
        ans=comp['choices'][0]['message']['content']
        err,raw=extract_code(ans)
        if err: continue
        opath=os.path.join(output_dir, f"op_{cid}.txt")
        rcode=raw + (pyspice_template.replace('[OP_PATH]',opath) if not args.ngspice else '')
        code_path = os.path.join(output_dir, f"code_{cid}.py")
        with open(code_path,'w') as cf: cf.write(rcode)
        eerr,serr,einfo,_=run_code(code_path)
        if eerr or serr:
            raw_code_error=serr or einfo
            continue
        answer=ans; break

    flog_name = f"log_{args.model}_p{tid}_n{args.num_per_task}.txt"
    with open(os.path.join(output_root, flog_name), 'a') as flog:
        flog.write(f"Chosen code_id={cid}, path={code_path}\n")

        netgen = os.path.join(output_dir, f"net_{cid}.py")
        with open(netgen,'w') as ngf: ngf.write(raw + output_netlist_template)
        subprocess.run(["python",netgen],check=True,text=True,stdout=subprocess.PIPE)
        npath = os.path.join(output_dir, f"net_{cid}.sp")
        open(npath,'w').write(subprocess.run(["python",netgen],stdout=subprocess.PIPE,text=True).stdout)

        if 'Opamp' in ttype or 'Amplifier' in ttype:
            netc=open(npath).read(); vn,vp=get_vin_name(netc,ttype)
            dcfile=os.path.join(output_dir, f"dc_{cid}.txt")
            dccode=raw + "\nsimulator=circuit.simulator()\n" + dc_sweep_template.replace('[IN_NAME]',vn).replace('[DC_PATH]',dcfile)
            dc_code_path=os.path.join(output_dir, f"dc_{cid}.py")
            open(dc_code_path,'w').write(dccode)
            subprocess.run(["python",dc_code_path],check=True,text=True)
            _,bv=get_best_voltage(dcfile)
            newcode=replace_voltage(raw,bv,vn,vp)
            with open(code_path,'w') as cf: cf.write(newcode)
            run_code(code_path)

        warn,msg=check_netlist(npath,opath,inp,outp,tid,ttype)
        if warn>0:
            flog.write(msg)
        else:
            ferr,fmsg=check_function(tid,code_path,ttype)
            if ferr: flog.write(fmsg)

    return 0





def get_retrieval(task, task_id):
    """
    
    """
    prompt = open('retrieval_prompt.md', 'r').read().replace('[TASK]', task)
    messages = [
        {"role": "system", "content": "You are an analog integrated circuits expert."},
        {"role": "user",   "content": prompt}
    ]
    subcircuits = [11]
    if args.retrieval:
        try:
            completion = send_api_chat(
                model       = args.model,
                messages    = messages,
                api_key     = args.api_key,
                temperature = args.temperature
            )
            answer = completion['choices'][0]['message']['content']
            # save retrieval output
            base_dir = args.model.replace('-', '').replace('turbo', '')
            os.makedirs(os.path.join(base_dir, f"p{task_id}"), exist_ok=True)
            with open(os.path.join(base_dir, f"p{task_id}", "retrieve.txt"), 'w') as f:
                f.write(answer)
            # parse list from markdown code block
            match = re.search(r".*?```.*?(.*?)```", answer, re.DOTALL)
            if match:
                subcircuits = eval(match.group(1))
        except Exception as e:
            print(f"Retrieval error: {e}")
    return subcircuits




def main():
    df = pd.read_csv('problem_set.tsv', sep='\t')

    # 只使用传入的 args.model，而不是循环所有模型
    mdl = args.model
    print(f"==={mdl}===")
    for _, r in df.iterrows():
        if r.Id != args.task_id:
            continue
        with open(f"log_{mdl}.txt", 'w') as flog:
            for i in range(args.num_of_done, args.num_per_task):
                subs = get_retrieval(r.Circuit, args.task_id) if r.Type in complex_task_type else None
                work(r.Circuit, r.Input.strip(), r.Output.strip(), args.task_id, i, None, r.Type, flog, subcircuits=subs)



    
if __name__ == "__main__":
    main()




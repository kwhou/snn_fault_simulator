# Copyright (c) 2022 Kuan-Wei Hou
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import torch

DEBUG = 0

# Get CPU/GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} device is used by SNN Fault Simulator.".format(device))

class Proc:
    def __init__(self, dut, nn_size, fault_list, test_algorithm):
        self.dut = dut.to(device)
        self.nn_size = nn_size
        self.fault_list = fault_list
        self.test_algorithm = test_algorithm
        
    def get_tf(self, spike):
        non_zero = spike > 0
        cum = non_zero.cumsum(axis=0)
        cum = (cum == 0).cumsum(axis=0)
        tf = cum[-1,:] + 1
        return tf
        
    def write_command(self, d):
        for row in range(self.nn_size[0]):
            for col in range(self.nn_size[1]):
                self.dut.synapse.write_synapse([row, col], d)

    def spike_command(self, tf_min, tf_max=False):
        for row in range(self.nn_size[0]):
            if not tf_max:
                input_spike = torch.zeros((tf_min, self.nn_size[0]))
            else:
                input_spike = torch.zeros((tf_max, self.nn_size[0]))
            input_spike[:,row] = 1
            output = self.dut(input_spike.to(device))
            tf = self.get_tf(output)
            if not tf_max:
                faulty = (tf <= tf_min).int().sum().item()
            else:
                faulty = ((tf < tf_min).int() + (tf > tf_max).int()).sum().item()
            if faulty:
                return True
        return False
        
    def get_fault_coverage(self, detected_fault_list, escaped_fault_list):
        total_fault = len(detected_fault_list) + len(escaped_fault_list)
        return float(len(detected_fault_list)) / total_fault
        
    def apply_test_algorithm(self, debug):
        detection = False
        for cmd in self.test_algorithm:
            if cmd[0] == 'W':
                if debug: print("Execute Command: {} {}".format(cmd[0], cmd[1]))
                self.write_command(int(cmd[1]))
            elif cmd[0] == 'S':
                if debug: print("Execute Command: {} {} {}".format(cmd[0], cmd[1], cmd[2]))
                faulty = self.spike_command(int(cmd[1]), int(cmd[2]))
                if faulty:
                    detection = True
                    break
            elif cmd[0] == 'SZ':
                if debug: print("Execute Command: {} {}".format(cmd[0], cmd[1]))
                faulty = self.spike_command(int(cmd[1]))
                if faulty:
                    detection = True
                    break
            elif cmd[0] == 'VTH':
                if debug: print("Execute Command: {} {}".format(cmd[0], cmd[1]))
                self.dut.set_vth(float(cmd[1]))
            elif cmd[0] == 'SWP':
                if debug: print("Execute Command: {} {}".format(cmd[0], cmd[1]))
                self.dut.set_swp(int(cmd[1]))
        return detection
        
    def test_fault_free(self):
        print("========== Fault-Free Simulation Start ========")
        detection = self.apply_test_algorithm(debug=0)
        if detection:
            print("Test Fail!\n")
        else:
            print("Test Pass!\n")

    def test(self):
        print("========== Fault Simulation Start    ==========")
        # Simulator output
        defected_fault_list = []
        escaped_fault_list = []
        
        for fault, depend_w, depend_swp in self.fault_list:
            for row in range(self.nn_size[0]):
                for col in range(self.nn_size[1]):
                    # set victim_list
                    print("Injecting Fault {} depend_w={} depend_swp={} location={}".format(fault, depend_w, depend_swp, [row, col]))
                    self.dut.synapse.set_victim_list(fault, depend_w, depend_swp, [row, col])
                    
                    # apply test algorithm
                    detection = self.apply_test_algorithm(debug=DEBUG)
                    
                    if detection:
                        print("Detected!\n")
                        defected_fault_list.append([fault, depend_w, depend_swp, [row, col]])
                    else:
                        print("Escaped!\n")
                        escaped_fault_list.append([fault, depend_w, depend_swp, [row, col]])
                        
        fault_coverage = self.get_fault_coverage(defected_fault_list, escaped_fault_list)
        
        with open('report.txt', 'w') as f:
            f.write("========== Detected Faults ==========\n")
            for fault in defected_fault_list:
                f.write("{} depend_w={} depend_swp={} location={}\n".format(fault[0], fault[1], fault[2], fault[3]))
                
            f.write("\n========== Escaped Faults ==========\n")
            for fault in escaped_fault_list:
                f.write("{} depend_w={} depend_swp={} location={}\n".format(fault[0], fault[1], fault[2], fault[3]))
                
            f.write("\nFault Coverage: {}\n".format(fault_coverage))
            
        print("Fault Coverage: {}\n".format(fault_coverage))

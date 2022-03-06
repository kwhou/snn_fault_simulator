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

import random
import torch
import torch.nn as nn

# Hardware Related Parameters
VR = 400.0              # Initial membrane potential of neuron
DV_MAX_POS = 24.57      # Maximum fault-free dV for weight=1
DV_MIN_POS = 2.45       # Minimum fault-free dV for weight=1
DV_MAX_NEG = -4.83      # Maximum fault-free dV for weight=-1
DV_MIN_NEG = -18.11     # Minimum fault-free dV for weight=-1
DV_FIF_POS = 25.0       # FIF dV for weight=1
DV_SIF_POS = 2.38       # SIF dV for weight=1
DV_FIF_NEG = -20.0      # FIF dV for weight=-1
DV_SIF_NEG = -4.54      # SIF dV for weight=-1

class Synapse(nn.Module):
    def __init__(self, array_size):
        super(Synapse, self).__init__()
        self.array_size = array_size
        self.nn_size = [array_size[0], array_size[1]//2]
        self.weight = nn.Parameter(torch.zeros(self.nn_size))
        self.fault_name = ''
        self.victim_list = []
        
    def initialize(self):
        self.weight[:,:] = 0
        self.fault_name = ''
        self.victim_list = []
        
    def write_synapse(self, addr, data):
        if data == 1:
            location = [addr[0], addr[1] * 2]
        elif data == -1:
            location = [addr[0], addr[1] * 2 + 1]
        else:
            location = 0
        
        if location in self.victim_list:
            if data == 1:
                if self.fault_name == 'SIF':
                    self.weight[addr[0]][addr[1]] = DV_SIF_POS
                elif self.fault_name == 'FIF':
                    self.weight[addr[0]][addr[1]] = DV_FIF_POS
            elif data == -1:
                if self.fault_name == 'SIF':
                    self.weight[addr[0]][addr[1]] = DV_SIF_NEG
                elif self.fault_name == 'FIF':
                    self.weight[addr[0]][addr[1]] = DV_FIF_NEG
            else:
                self.weight[addr[0]][addr[1]] = 0
        else:
            if data == 1:
                self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_POS, DV_MAX_POS)
            elif data == -1:
                self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_NEG, DV_MAX_NEG)
            else:
                self.weight[addr[0]][addr[1]] = 0
        
    def set_victim_list(self, fault, location):
        if location[0] >= self.array_size[0] or \
            location[1] >= self.array_size[1]:
            print("Error: {} out of array size.".format(location))
        else:
            self.victim_list = []
            if fault == 'SIF_bc':
                self.fault_name = 'SIF'
                self.victim_list.append(location)
            elif fault == 'SIF_r':
                self.fault_name = 'SIF'
                row = location[0]
                for col in range(self.array_size[1]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_ra':
                self.fault_name = 'SIF'
                row = location[0]
                for col in range(location[1], self.array_size[1]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_c':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(self.array_size[0]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_cb':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(0, location[0]+1):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_ca':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(location[0], self.array_size[0]):
                    self.victim_list.append([row, col])
            elif fault == 'FIF_bc':
                self.fault_name = 'FIF'
                self.victim_list.append(location)
            else:
                print("Error: {} not recognized as a supported fault.".format(fault))
            
    def forward(self, input_spike):
        return torch.matmul(input_spike, self.weight)

class Layer(nn.Module):
    def __init__(self, array_size):
        super(Layer, self).__init__()
        self.array_size = array_size
        self.nn_size = [array_size[0], array_size[1]//2]
        self.swp = 0
        self.vth = 500.0
        self.synapse = Synapse(array_size)
        
    def initialize(self):
        self.swp = 0
        self.vth = 500.0
        self.synapse.initialize()
        
    def set_swp(self, swp):
        self.swp = swp
        
    def set_vth(self, vth):
        self.vth = vth
        
    def fire(self, v):
        if self.swp == 0:
            out = torch.where(v >= self.vth, torch.ones_like(v), torch.zeros_like(v))
        else:
            out = torch.where(v <= self.vth, torch.ones_like(v), torch.zeros_like(v))
        return out
        
    def reset(self, sout, v):
        return torch.where(torch.eq(sout, 1), torch.zeros_like(v), v)
    
    def forward(self, in_spike):
        # in_spike size: (num_of_tick, num_of_input)
        num_of_tick = in_spike.size()[0]
        
        # calculate dv
        dv = self.synapse(in_spike)
        
        # generate output spike
        olist = []
        for t in range(num_of_tick):
            # integrate
            if t == 0:
                v = dv[t,:] + VR
            else:
                v = v + dv[t,:]
            # fire
            output = self.fire(v)
            olist.append(output)
            # reset
            v = self.reset(output, v)
        output = torch.stack(olist, dim=0)
        
        return output

# model = Layer([array_size[0], array_size[1]])

# model.synapse.set_victim_list('FIF_bc', [1,2])
# print(model.synapse.victim_list)

# model.synapse.write_synapse([1,1], 1)
# print(model.synapse.weight[1,1])

# output = model(torch.ones(5,array_size[0]))
# print(output)
# print(output.size())

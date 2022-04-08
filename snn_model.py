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
VR = 400.0                  # Initial membrane potential of neuron
DV_MAX_WPOS = 24.57         # Maximum fault-free dV for weight=1
DV_MIN_WPOS = 2.45          # Minimum fault-free dV for weight=1
DV_MAX_WNEG = -4.83         # Maximum fault-free dV for weight=-1
DV_MIN_WNEG = -18.11        # Minimum fault-free dV for weight=-1
DV_MAX_WZERO = 8.2          # Maximum fault-free dV for weight=0
DV_MIN_WZERO = -6.1         # Minimum fault-free dV for weight=0
DV_FIF_WPOS = 25.0          # FIF dV for weight=1
DV_SIF_WPOS = 2.38          # SIF dV for weight=1
DV_FIF_WNEG = -20.0         # FIF dV for weight=-1
DV_SIF_WNEG = -4.54         # SIF dV for weight=-1
DV_FIF_WZERO_SWP0 = 8.34    # FIF dV for weight=0 and SWP=0
DV_FIF_WZERO_SWP1 = -6.25   # FIF dV for weight=0 and SWP=1

class Synapse(nn.Module):
    def __init__(self, nn_size):
        super(Synapse, self).__init__()
        self.nn_size = nn_size
        self.weight = nn.Parameter(torch.zeros(self.nn_size))
        self.fault_name = ''
        self.victim_list = []
        self.depend_w = ''
        self.depend_swp = ''
        
    def initialize(self):
        self.weight[:,:] = 0
        self.fault_name = ''
        self.victim_list = []
        self.depend_w = ''
        self.depend_swp = ''
        
    def write_synapse(self, addr, data):
        if addr in self.victim_list and data == self.depend_w:
            if data == 1:
                if self.fault_name == 'SIF':
                    self.weight[addr[0]][addr[1]] = DV_SIF_WPOS
                elif self.fault_name == 'FIF':
                    self.weight[addr[0]][addr[1]] = DV_FIF_WPOS
            elif data == -1:
                if self.fault_name == 'SIF':
                    self.weight[addr[0]][addr[1]] = DV_SIF_WNEG
                elif self.fault_name == 'FIF':
                    self.weight[addr[0]][addr[1]] = DV_FIF_WNEG
            else:
                if self.fault_name == 'FIF':
                    if self.depend_swp:
                        self.weight[addr[0]][addr[1]] = DV_FIF_WZERO_SWP1
                    else:
                        self.weight[addr[0]][addr[1]] = DV_FIF_WZERO_SWP0
                else:
                    self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_WZERO, DV_MAX_WZERO)
        else:
            if data == 1:
                self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_WPOS, DV_MAX_WPOS)
            elif data == -1:
                self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_WNEG, DV_MAX_WNEG)
            else:
                self.weight[addr[0]][addr[1]] = random.uniform(DV_MIN_WZERO, DV_MAX_WZERO)
        
    def set_victim_list(self, fault, depend_w, depend_swp, location):
        if location[0] >= self.nn_size[0] or \
            location[1] >= self.nn_size[1]:
            print("Error: {} out of array size.".format(location))
        else:
            self.victim_list = []
            self.depend_w = depend_w
            self.depend_swp = depend_swp
            if fault == 'SIF_bc':
                self.fault_name = 'SIF'
                self.victim_list.append(location)
            elif fault == 'SIF_r':
                self.fault_name = 'SIF'
                row = location[0]
                for col in range(self.nn_size[1]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_ra':
                self.fault_name = 'SIF'
                row = location[0]
                for col in range(location[1], self.nn_size[1]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_c':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(self.nn_size[0]):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_cb':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(0, location[0]+1):
                    self.victim_list.append([row, col])
            elif fault == 'SIF_ca':
                self.fault_name = 'SIF'
                col = location[1]
                for row in range(location[0], self.nn_size[0]):
                    self.victim_list.append([row, col])
            elif fault == 'FIF_bc':
                self.fault_name = 'FIF'
                self.victim_list.append(location)
            elif fault == 'FIF_ca':
                self.fault_name = 'FIF'
                col = location[1]
                for row in range(location[0], self.nn_size[0]):
                    self.victim_list.append([row, col])
            else:
                print("Error: {} not recognized as a supported fault.".format(fault))
            
    def forward(self, input_spike):
        return torch.matmul(input_spike, self.weight)

class Layer(nn.Module):
    def __init__(self, nn_size):
        super(Layer, self).__init__()
        self.nn_size = nn_size
        self.swp = 0
        self.vth = 500.0
        self.synapse = Synapse(nn_size)
        
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
        return torch.where(torch.eq(sout, 1), torch.full_like(v, VR), v)
    
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

if __name__ == '__main__':
    array_size = [2, 2]
    model = Layer(array_size)
    
    # model.synapse.set_victim_list('SIF_bc', 1, 0, [1,1])
    model.synapse.set_victim_list('FIF_bc', 1, 0, [1,1])
    print(model.synapse.fault_name)
    print(model.synapse.victim_list)
    print(model.synapse.depend_w)
    print(model.synapse.depend_swp)
    
    model.synapse.write_synapse([1,1], 1)
    print(model.synapse.weight)
    
    output = model(torch.ones(41,array_size[0]))
    print(output)

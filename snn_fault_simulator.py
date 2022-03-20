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

import read_input
import snn_model
import test_procedure

# Read simulator input
array_size = read_input.read_array_size("array_size.txt")
fault_list = read_input.read_fault_list("fault_list.txt")
test_algorithm = read_input.read_test_algorithm("test_algorithm.txt")

# Verify the array size
if array_size[1] % 2:
    print("Error: number of columns must be an even number.")
    exit()
nn_size = [array_size[0], array_size[1]//2]

# SNN model initialization
model = snn_model.Layer(nn_size)

# Create and link the test procedure
proc = test_procedure.Proc(model, nn_size, fault_list, test_algorithm)

# Run fault simulation
proc.test()

# Run fault-free simulation
model.initialize()
proc.test_fault_free()

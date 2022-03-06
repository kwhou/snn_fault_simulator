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

def read_array_size(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            arg = line.split()
            if arg[0][0] != '#':
                row = int(arg[0])
                col = int(arg[1])
                break
    return [row, col]

def read_fault_list(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        fault_list = []
        for line in lines:
            arg = line.split()
            if arg[0][0] != '#':
                fault_list.append(arg[0])
    return fault_list

def read_test_algorithm(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        test_algorithm = []
        for line in lines:
            arg = line.split()
            if arg[0][0] != '#':
                test_algorithm.append(arg)
    return test_algorithm

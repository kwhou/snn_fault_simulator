# SNN Fault Simulator

![NTHU LARC Logo](images/nthu_larc_logo.png?raw=true)

## Usage

1. Set the 1T1R array size in the file "array_size.txt".
```
# row col
  256   6
```

2. Set the fault list in the file "fault_list.txt".
```
# fault
SIF_bc
SIF_r
SIF_ra
SIF_c
SIF_cb
SIF_ca
FIF_bc
```

3. Set the test algorithm in the file "test_algorithm.txt".
```
# operation
SWP 0
VTH 500
W 1
S 5 41
SWP 1
VTH 300
W -1
S 6 21
```

4. Run fault simulation.
```
$ python snn_fault_simulator.py
```

5. The simulation result can be found in file "report.txt".

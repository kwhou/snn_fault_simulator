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
# fault  depend_w  depend_swp
SIF_bc      1           0
SIF_bc     -1           1
SIF_r       1           0
SIF_r      -1           1
SIF_ra      1           0
SIF_ra     -1           1
```

3. Set the test algorithm in the file "test_algorithm.txt".
```
# command
SWP 0
VTH 500
W 1
S 5 41
W 0
SZ 12
SWP 1
VTH 300
SZ 16
W -1
S 6 21
```

4. Run fault simulation.
```
$ python snn_fault_simulator.py
```

5. The simulation result can be found in file "report.txt".
```
========== Detected Faults ==========
SIF_bc depend_w=1 depend_swp=0 location=[0, 0]
SIF_bc depend_w=1 depend_swp=0 location=[0, 1]
SIF_bc depend_w=1 depend_swp=0 location=[0, 2]
...

========== Escaped Faults ==========

Fault Coverage: 1.0
```

## Citation

If you use this code in your work, please cite the following paper.
K.-W. Hou, H.-H. Cheng,  C. Tung, C.-W. Wu and J.-M. Lu, "Fault Modeling and Testing of Memristor-Based Spiking Neural Networks", in Proc. IEEE Int. Test Conf. (ITC), Anaheim, Sept. 2022.

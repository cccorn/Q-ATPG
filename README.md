# Q-ATPG
The implementation of “Automatic Test Pattern Generation for Robust Quantum Circuit Testing”.

### Requirements
- Python 3
- Qiskit
- CVXPY

### Installation
1. Clone this repo
```
git clone https://github.com/cccorn/Q-ATPG.git
cd Q-ATPG
```
2. Install the python packages:
```
pip install qiskit
pip install cvxpy
```
3. Create directories:
```
mkdir cache results
```
4. Precompute and cache the data:
```
python gen_cache.py
```

### Demo
```
python demo_genSPD_samp.py
```
You can modify the configurations in `lib/config.py` to change the parameters.

```
Copyright 2021 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```

# Python HITMIX

Computation of hitting time moments using a linear algebraic approach,
as defined in the following paper:

- A Deterministic Hitting-Time Moment Approach to Seed-set Expansion over a Graph
- A. Foss, R.B. Lehoucq, Z.W. Stuart, J.D.Tucker, J.W. Berry
- https://arxiv.org/pdf/2011.09544.pdf

## Current Capabilities Available
* Computation of first two hitting time momments
 
## Quick Start

### Install
* User: ```python3 setup.py install```
* Developer: ```python3 setup.py develop```

### Testing
```
# requires `pytest`
python -m pytest --disable-pytest-warnings
```

### Coverage Testing
```
# requires `pytest-cov`
pytest --cov=hitmix  tests/ --cov-report=html
# output can be accessed via htmlcov/index.html
```

### Documentation
```
# requires `sphinx`
sphinx-build ./docs/source ./docs/build
# output can be accessed via docs/build/html/index.html
```

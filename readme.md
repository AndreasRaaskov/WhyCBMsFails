# Why Concept bottleneck models don't learn as indented

This work will document how concept bottleneck models produces deceptive expandability.



## Setup

### Install virtual environment

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

To deactivate the virtual environment when done:
```bash
deactivate
```

### Get dataset
Dataset is downloaded from the original CUP paper to ensure original train test validation split.

Original CUP dataset
https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5

Majority voted labels
https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683

Models
Concept model: https://worksheets.codalab.org/bundles/0xcaed16afacbe4d3e9146b7c3fac15032


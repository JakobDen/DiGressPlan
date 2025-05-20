# DiGress: Discrete Denoising diffusion models for room plan generation
The code adjusted for https://github.com/HanHan55/Graph2plan dataset.
![graph_22](https://github.com/user-attachments/assets/61a7604d-c511-4381-a5bf-ac632b56e7c2)
![graph_17](https://github.com/user-attachments/assets/9365db79-4658-42e5-b8c5-f0cf7ac6fa8e)
Class SpectreGraphDataset fullu rewritten. Also, I've changed the model so that it generates directed graphs.
## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
  - `conda activate digress`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the discrete model: `python3 main.py`

## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```

# pyDDP
Use PyTorch's distributed data parallel to train a simple model in a multi-GPU environment. The example includes demonstration of using gradient accumulation in training along with using mix precision. Hyper params are included in the __main__ function of the script, use `python ./Simple_DDP.py` to start it or `python ./Simple_DDP_mpbf16.py` to start the mix precision example.

# OnlineLDS

Source code for the AAAI 2019 paper "On-Line Learning of Linear Dynamical Systems: Exponential Forgetting in Kalman Filters" (https://arxiv.org/abs/1809.05870). If you use this code, please cite the paper as::

    @inproceedings{kozdoba2018,
     title={On-Line Learning of Linear Dynamical Systems: Exponential Forgetting in Kalman Filters},
      author={Kozdoba, Mark and Marecek, Jakub and Tchrakian, Tigran and Mannor, Shie},
      booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)},
      note={arXiv preprint arXiv:1809.05870},
      year={2019}
    }

Running experiments.py recreates the plots used in the final 8-page version of the paper.
Further calls explained therein produce further illuminating plots, which did not make it 
into the final 8-page version of the paper. Some of these require data from Prof. Steve
Hoi to be placed in ./OARIMA_code_data/ -- please download these separately from http://oarima.stevenhoi.org/ 

To reuse the algorithms introduce by ourselves, see onlinelds.py for an implementation. 

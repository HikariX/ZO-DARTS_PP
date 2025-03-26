#  ZO-DARTS++

## Basic Info

This code repository is prepared for *ZO-DARTS++: An Efficient and Size-Variable Zeroth-Order Neural Architecture Search Algorithm* and *An Eﬃcient Neural Architecture Search Model for Medical Image Classification (ZO-DARTS+)*. The former one is under review for one journal, and the latter on is published on European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning 2024. We implement our algorithm and other algorithms used for comparison on NAS-Bench-201 based on the repository https://github.com/D-X-Y. We express our gratitude to the owner of this repository. We also use MedMNIST datasets.

## Important code addition & change

### ./exps/NAS-Bench-201-algos/

#### Codes for DARTS-style methods on NAS-Bench-201:

iDARTS.py, MiLeNAS.py, PCDARTS.py, PCDARTS-OURS.py, DARTS+.py

#### Codes for both ZO-DARTS++ and ZO-DARTS+ experiments:

files with the "-med" suffix.

1. **ZO-DARTS_SPARSE.py**: the ZO-DARTS using sprasemax function.
2. **ZO-DARTS_SPARSEMIX.py**: extends 1 with the mixed kernel sizes.
3. **ZO-DARTS_SPARSEMIXEXIT.py**: extends 2 with the mixed cell numbers in each stage.
4. **ZO-DARTS_SPARSE_anneal.py**: extends 1 using sprasemax function and temperature annealing scheme.
5. **ZO-DARTS_SPARSE_anneal_med.py**: extends 4 on medMNIST dataset. **Used for ZO-DARTS+**.
6. **ZO-DARTS_SPARSE_anneal_med_cellN.py**: extends 5 on medMNIST dataset, and try to have different structures for cells in different stages after certain epochs.
7. **ZO-DARTS_SPARSE_anneal_med_penalty.py**: extends 5 on medMNIST dataset, and try to have resource constraint after certain epochs.
8. **ZO-DARTS_SAMVR.py**: extends both 6 and 7, also combined 3 to have varied structures. **Used for ZO-DARTS++**.
9. **ZO-DARTS_SAVR.py**: based on 8, but used for cifar-like dataset.

### ./xautodl/models/cell_searchs

#### Codes for DARTS-style methods on NAS-Bench-201:

search_cells_pcdarts.py, search_model_pcdarts.py

#### Codes for both ZO-DARTS++ and ZO-DARTS+ experiments:

1. **search_model_sparsezo.py**: for the ZO-DARTS using sprasemax function.
2. **search_model_sparsezomix.py**: extends 1 with the mixed kernel sizes.
3. **search_model_sparsezomixexit.py**: extends 2 with the mixed cell numbers in each stage.
4. **search_model_sparsezo_anneal.py**: extends 1 with temperature annealing scheme. **Used for ZO-DARTS+**.
5. **search_model_sparsezo_anneal_cellN.py**: extends 4, try to have different structures for cells in different stages after certain epochs.
6. **search_model_ZO_SMEA.py**: extends 5, also combined 3 to have varied structures.  **Used for ZO-DARTS++**. 

### ./xautodl/models

**cell_operations.py**: add some variable operations specified for ZO-DARTS++.

### ./xautodl/datasets

1. **MedMNIST.py**: a dedicated file for being the loader of medMNIST data.
2. **get_dataset_with_transform.py**: added some code pieces for handling medMNIST data.

### ./retrain

To be continued

## Main body of ZO-DARTS++

We implement our ZO-DARTS++ based on DARTS-V2.py. Important functions are listed as below. Please see ZO-DARTS_SAMVR.py

1. **kernelV_param_num**: calculates the parameter numbers for different kernels.
2. **count_reduc_num**: calculation of parameter numbers for reduction cells.
3. **resource_calculator**: calculates the resource consumption.
4. **_generate_z**: used for generating direction vector (u).
5. **_prepare_distrubance**: used to generate the disturbed model weights and architecture parameters.
6. **_backward_step_ours**: defined for calculating approximated gradients of $\alpha$.
7. **_backward_step_ours_mixed_penalty**: defined for calculating approximated gradients of $\alpha$ under resource constraint.
8. **search_func_ours_F**: forms the framework of the searching process defined in Algorithm 1 and invokes functions mentioned above.

## How to execute

Use the command below can start the search process of ZO-DARTS++:

```
python ZO-DARTS_SAMVR.py --rand_seed ${seed} --data_path ${path} --dataset ${dataset} --size ${s} --batch_size ${batch} --save_dir ${dir}
```

Running this command starts ZO-DARTS++.

## Acknowledgements

The skeleton codebase in this repository was adapted from NAS-Bench-201[1].

[1] X. Dong and Y. Yang, “Nas-bench-201: Extending the scope of reproducible neural architecture search,” in ICLR, 2020.

for b in 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000
    do
        nohup python -u retrain_sparse_cellN.py --rand_seed 1 --budget {$b} --dataset OrgansMNIST>./ParetoFront/organs_{$b}_seed1.log 2>&1
    done
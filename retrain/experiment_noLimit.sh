for seed in 1 2 3
do
    for r in 1 2 3
    do
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset OrganCMNIST > ./result_Full/organc_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset OrganAMNIST > ./result_Full/organa_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset OrganSMNIST > ./result_Full/organs_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset OCTMNIST > ./result_Full/oct_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset TissueMNIST > ./result_Full/tissue_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset PneumoniaMNIST > ./result_Full/pneumonia_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset BreastMNIST > ./result_Full/breast_seed${seed}_round${r}.log 2>&1 &

        nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset PathMNIST > ./result_Full/path_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset DermaMNIST > ./result_Full/derma_seed${seed}_round${r}.log 2>&1 &
        # nohup python -u retrain_sampling.py --rand_seed ${seed} --dataset BloodMNIST > ./result_Full/blood_seed${seed}_round${r}.log 2>&1 &
    done
done
for seed in 1 2 3
do
    for b in 1 2 3
    do
        for r in 1 2 3
        do
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset OrganCMNIST > ./result_percentile/organc_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset OrganAMNIST > ./result_percentile/organa_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset OrganSMNIST > ./result_percentile/organs_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset OCTMNIST > ./result_percentile/oct_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset TissueMNIST > ./result_percentile/tissue_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset PneumoniaMNIST > ./result_percentile/pneumonia_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset BreastMNIST > ./result_percentile/breast_constraint${b}_seed${seed}_round${r}.log 2>&1 &

            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset PathMNIST > ./result_percentile/path_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset DermaMNIST > ./result_percentile/derma_constraint${b}_seed${seed}_round${r}.log 2>&1 &
            nohup python -u retrain_sampling.py --rand_seed ${seed} --budget ${b} --dataset BloodMNIST > ./result_percentile/blood_constraint${b}_seed${seed}_round${r}.log 2>&1 &
        done
    done
done
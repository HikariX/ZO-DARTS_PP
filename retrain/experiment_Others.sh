for seed in 1
do
    for m in DARTS MileNAS ZO DARTSAER ZOP
    do
        for r in 1 2 3
        do
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset OrganAMNIST >./NewExp2025/result_Others/${m}_organa_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset OrganCMNIST >./NewExp2025/result_Others/${m}_organc_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset OrganSMNIST >./NewExp2025/result_Others/${m}_organs_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset PneumoniaMNIST >./NewExp2025/result_Others/${m}_pneumonia_seed${seed}_round${r}.log 2>&1 &
            nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset OCTMNIST >./NewExp2025/result_Others/${m}_oct_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset BreastMNIST >./NewExp2025/result_Others/${m}_breast_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset TissueMNIST >./NewExp2025/result_Others/${m}_tissue_seed${seed}_round${r}.log 2>&1 &
            
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset BloodMNIST >./NewExp2025/result_Others/${m}_blood_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset DermaMNIST >./NewExp2025/result_Others/${m}_derma_seed${seed}_round${r}.log 2>&1 &
            # nohup python -u retrain_sampling_others.py --GPU 0 --method ${m} --rand_seed ${seed} --dataset PathMNIST >./NewExp2025/result_Others/${m}_path_seed${seed}_round${r}.log 2>&1 &
        done
    done
done

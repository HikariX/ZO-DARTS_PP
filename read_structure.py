import glob, os, sys

def read_structure(seed=None, dataset=None, budget=None):
    path = "/root/ZO-DARTS_light_82/exps/NAS-Bench-201-algos/ParetoFront/ZO_SAP+" + str(seed) + "_" + dataset[:-5].lower() + "+" + str(budget) + "_test/"
    file = open(glob.glob(path + "*.log")[0], "r")
    # 打开并读取每个文件的内容
    content = file.readlines()
    structure_recorder = ''
    exit_recorder = []
    flag0 = 0
    flag1 = 0
    for line in content:
        if "The 049-050-th epoch" in line:
            flag0 = 1
        if flag0 and "arch-parameters-exit :" in line:
            flag1 = 1

        if "last-geno" in line:
            structure_recorder = line
        elif flag0 and flag1:
            exit_recorder.append(line)
    structures = structure_recorder.split('Structure(4 nodes with ')[1:]
    structures[0] = structures[0][:-3] # Delete excessive brackets
    structures[1] = structures[1][:-3]
    structures[2] = structures[2][:-4]

    exit_recorder = exit_recorder[1:4]
    exit_recorder[0] = exit_recorder[0][8:]
    exit_recorder[-1] = exit_recorder[-1][:-3]
    exit_recorder = [eval(i.split(',\n')[0].strip()) for i in exit_recorder ]
    cell_list = [i.index(max(i)) + 1 for i in exit_recorder]
    return structures, cell_list

budget_list = [i * 100000 for i in range(1, 13)]

for seed in range(1, 5):
    for budget in budget_list:
        structures, cell_list = read_structure(seed, 'OrganSMNIST', budget)
        with open('./structure_list_seed' + str(seed) + '.txt', 'a+', encoding='utf-8') as file:
            file.write('budget:%s\n' % str(budget))
            for item in structures:
                file.write("%s\n" % item)
            for item in cell_list:
                file.write("%s\n" % item)

    # # 读取文件
    # with open('./structure_list_seed1.txt', 'r', encoding='utf-8') as file:
    #     read_list = file.read().splitlines()

import os
import numpy as np
import pandas as pd


def summarise(root='./'):
    results =[]
    metrics = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    
    arch_names = os.listdir(root)
    for arch_name in arch_names:
        if '.py' in arch_name or 'Augmentation' in arch_name:
            continue
        dir_path = os.path.join(root, arch_name)
        for file_name in os.listdir(dir_path):
            setting = os.path.basename(file_name)
            res = []
            res.append(arch_name)
            res.append(setting)

            with open(os.path.join(dir_path, setting), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'F1' in line:
                        res.append('{:.4f}'.format(float(line.strip('\n').split(' ')[-1])))
                        continue
                    if 'IoU_mean' in line:
                        res.append('{:.4f}'.format(float(line.strip('\n').split(' ')[-1])))
            results.append(res)
    #         break
            
    #     break
    # print(results)
    columns = ['Architecture', 'Setting']
    for metric in metrics:
        columns.append(metric + '-mDice')
        columns.append(metric + '-mIoU')
    df = pd.DataFrame(results, columns=columns)
    df.to_excel('./summary_results.xlsx')

if __name__ == '__main__':
    summarise()
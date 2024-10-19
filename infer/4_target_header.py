import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='validation.')
parser.add_argument('--result_file', type=str, help=' ')
parser.add_argument('--header_file', type=str, help=' ')
parser.add_argument('--thresholds', type=str, help=' ')
parser.add_argument('--target_header_file', type=str, help=' ')
parser.add_argument('--target_result_file', type=str, help=' ')
parser.add_argument('--ratio_file_path', type=str, help=' ')

args = parser.parse_args()


result_file = args.result_file
header_file = args.header_file

thresholds = args.thresholds
target_header_file = args.target_header_file
target_result_file = args.target_result_file
ratio_file_path = args.ratio_file_path


result_data = np.loadtxt(result_file)
header_data = np.loadtxt(header_file, dtype=str)
thresholds=np.array([0.1,0.15,0.2])

print(thresholds)
for threshold in thresholds:



    print(threshold)
    threshold=np.array(threshold)
    indices = np.where(result_data < threshold)[0]

    target_headers = header_data[indices]
    target_result = result_data[indices]


    target_header_file_threshold = target_header_file.replace('.txt', f'_{str(threshold).replace(".", "")}.txt')
    target_result_file_threshold = target_result_file.replace('.txt', f'_{str(threshold).replace(".", "")}.txt')

    np.savetxt(target_header_file_threshold, target_headers, fmt='%s')
    np.savetxt(target_result_file_threshold, target_result, fmt='%s')



print('begin cal')


indices = np.where(result_data <1)[0]


target_headers = header_data[indices]
target_result=result_data[indices]



likelihood_1 = target_result
likelihood_1 = likelihood_1.astype(np.float32)
likelihood_2 = 1 - likelihood_1


max_sum = -np.inf
best_gap_index = -1
gaps = np.linspace(0, 1, 1001)

print('begin for')

for i, gap in enumerate(gaps):
    score = np.array([gap, 1 - gap])  

    val = np.log10(score[0] * likelihood_1 + score[1] * likelihood_2)
    current_sum = np.sum(val)
    

    if current_sum > max_sum:
        max_sum = current_sum
        best_gap_index = i


os.makedirs(os.path.dirname(ratio_file_path), exist_ok=True)
best_gap = gaps[best_gap_index]

print(f"Contamination Ratio: {best_gap} (0-1), with sum: {max_sum}")

with open(ratio_file_path, 'w') as result_file:
    result_file.write(f"Contamination Ratio: {best_gap} (0-1), with sum: {max_sum}")
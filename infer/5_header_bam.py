import pysam
import numpy as np
import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='validation.')
parser.add_argument('--original_bam_file', type=str, help=' ')
parser.add_argument('--processed_bam_dir', type=str, help=' ')
parser.add_argument('--processed_bam_files', type=str, help=' ')
parser.add_argument('--thresholds', type=str, help=' ')
parser.add_argument('--target_header_file', type=str, help=' ')

args = parser.parse_args()


original_bam_file = args.original_bam_file
processed_bam_dir = args.processed_bam_dir



bam_files_list = args.processed_bam_files.split(',')


processed_bam_files = {}
for item in bam_files_list:
    threshold, path = item.split(':')
    processed_bam_files[float(threshold)] = path


thresholds=np.array([0.1,0.15,0.2])

target_header_file = args.target_header_file

os.makedirs(processed_bam_dir, exist_ok=True)

for threshold in thresholds:
    
    target_header_file_threshold = target_header_file.replace('.txt', f'_{str(threshold).replace(".", "")}.txt')
    processed_bam_file = processed_bam_files[threshold]
    processed_bam_file = processed_bam_file.replace("'", "")
    processed_bam_file = processed_bam_file.replace(" ", "")

    target_headers = np.loadtxt(target_header_file_threshold, dtype=str)
    target_headers_set = set(target_headers)

 
    with pysam.AlignmentFile(original_bam_file, 'rb') as original_bam, \
         pysam.AlignmentFile(processed_bam_file, 'wb', header=original_bam.header) as processed_bam:
        print(f'Processing with threshold {threshold}')
        for read in original_bam:
            if read.query_name in target_headers_set:
                processed_bam.write(read)

    bed_file = processed_bam_file.replace('.bam', '.bed')
    cmd = f"bamToBed -i {processed_bam_file} > {bed_file}"
    subprocess.run(cmd, shell=True, check=True)
    print(f'Converted {processed_bam_file} to {bed_file}')





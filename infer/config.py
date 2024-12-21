import os


data_dir = '/Users/zhenyizhang/Desktop/Deep_niPGT/DECENT/test/data/'


for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.rmdup.bam'):
                original_bam_path = os.path.join(folder_path, file_name)

                sample_id = file_name.split('.')[0]  
                sample_key = sample_id.split('_')[1]  


                new_bam_path = original_bam_path.replace('.rmdup.bam', '_id.bam')
                origin_bam_dir = folder_path + '/'
                reads_dir = f'/Users/zhenyizhang/Desktop/Deep_niPGT/DECENT/test/result/reads/{sample_key}/'
                score_dir = f'/Users/zhenyizhang/Desktop/Deep_niPGT/DECENT/test/result/score/{sample_key}/'
                processed_bam_dir = f'/Users/zhenyizhang/Desktop/Deep_niPGT/DECENT/test/result/processed_bam/{sample_key}/'


                ratio_file_path = processed_bam_dir + 'ratio.txt'


                thresholds = [0.1, 0.15, 0.2]


            
                processed_bam_files = {
                    threshold: processed_bam_dir + f'process_{sample_key}_{str(threshold).replace(".", "")}.bam'
                    for threshold in thresholds
                }

                processed_bam_files_str = [f"{key}: '{value}'" for key, value in processed_bam_files.items()]


                processed_bam_files_str_combined = ', '.join(processed_bam_files_str)


                file_dir = reads_dir  
                store_dir = score_dir  
                result_file = score_dir + f'result_{sample_id}_id.reads.txt' 
                header_file = score_dir + f'header_{sample_id}_id.reads.txt'  
                target_header_file = score_dir + f'target_header_file.txt'  
                target_result_file = score_dir + f'target_result_file.txt'  
                original_bam_file = new_bam_path  

                print(f"{original_bam_path}|{new_bam_path}|{origin_bam_dir}|{reads_dir}|{file_dir}|{store_dir}|{result_file}|{header_file}|{thresholds}|{target_header_file}|{target_result_file}|{ratio_file_path}|{original_bam_file}|{processed_bam_dir}|{processed_bam_files_str_combined}|{thresholds}|{target_header_file}")
    
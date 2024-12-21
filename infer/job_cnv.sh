#!/bin/bash


file_name="$1"


python config.py | while IFS='|' read -r original_bam_path new_bam_path origin_bam_dir reads_dir file_dir store_dir result_file header_file thresholds target_header_file target_result_file ratio_file_path original_bam_file processed_bam_dir processed_bam_files thresholds2 target_header_file2; do

    if [[ "$original_bam_path" != *"$file_name"* ]]; then
        continue  
    fi
    

    echo "File Name: $original_bam_path"

    python 1_id_bam.py --original_bam_path "$original_bam_path" --new_bam_path "$new_bam_path"
    python 2_extract_header.py --origin_bam_dir "$origin_bam_dir" --reads_dir "$reads_dir"
    python 3_new_traning_header_predict_tensor.py --file_dir "$file_dir" --store_dir "$store_dir"
    python 4_target_header.py --result_file "$result_file" --header_file "$header_file" --thresholds "$thresholds" --target_header_file "$target_header_file" --target_result_file "$target_result_file" --ratio_file_path "$ratio_file_path"
    python 5_header_bam.py --original_bam_file "$original_bam_file" --processed_bam_dir "$processed_bam_dir" --processed_bam_files "$processed_bam_files" --thresholds "$thresholds2"  --target_header_file "$target_header_file2"


    if [ "$run_once" = true ]; then
        break
    fi
done

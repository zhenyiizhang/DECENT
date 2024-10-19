import pysam
import argparse


parser = argparse.ArgumentParser(description='Process BAM files.')
parser.add_argument('--original_bam_path', type=str, help='Path to the original BAM file')
parser.add_argument('--new_bam_path', type=str, help='Path to the new BAM file')
args = parser.parse_args()

original_bam_path = args.original_bam_path
new_bam_path = args.new_bam_path


print('begin read bam')
original_bam = pysam.AlignmentFile(original_bam_path, 'rb')

print('begin create')
new_bam = pysam.AlignmentFile(new_bam_path, 'wb', header=original_bam.header)

print('begin for')

for count, read in enumerate(original_bam):

    new_id = f'{count:08d}'

    read.query_name = new_id

    new_bam.write(read)


original_bam.close()
new_bam.close()

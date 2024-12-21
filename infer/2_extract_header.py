import pysam
import numpy as npos
from collections import Counter
import random
import os
import os.path
import re
import sys
import codecs
import collections

import argparse
import pysam


parser = argparse.ArgumentParser(description='Extract headers from BAM files.')
parser.add_argument('--origin_bam_dir', type=str, help='Directory containing original BAM files')
parser.add_argument('--reads_dir', type=str, help='Directory to store reads files')
args = parser.parse_args()


origin_bam_dir = args.origin_bam_dir
reads_dir = args.reads_dir






os.makedirs(reads_dir, exist_ok=True)


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

if __name__ == '__main__':
    files = file_name(origin_bam_dir)
    for file in files:
        if ('id.bam' in file) and ('log' not in file):
            input = pysam.AlignmentFile(origin_bam_dir + file, 'rb')
            file = file[0:len(file)-4]
            output = open(reads_dir + file+'.reads', 'w')
            for line in input:
                sig = line.cigarstring
                if 'I' in sig:
                    continue
                name = line.reference_name
                start = str(line.reference_start)
                end = str(line.reference_end)
                read = line.get_tag('XG')
                if 'N' in read:
                    continue
                methy = line.get_tag('XM')
                length = len(read)
                read = read[3:length - 3]
                methylation = ''
                for i in range(len(methy)):
                    if methy[i] == 'X':
                        if i < (len(methy)-1):
                            if (read[i] == 'C') & (read[i+1] == 'G'):
                                methylation = methylation + '1'
                            else:
                                methylation = methylation + '0'
                        elif i == (len(methy)-1):
                            if read[i] == 'C':
                                methylation = methylation + '1'
                            else:
                                methylation = methylation + '0'
                        else:
                            methylation = methylation + '0'
                    else:
                        methylation = methylation+'0'
                if len(methylation) < 71:
                    continue
                else:
                 
                    output.write(line.query_name + '\t' + name + '\t' + start + '\t' + read[5:71] + '\t' + methylation[5:71] + '\n')
            output.close()

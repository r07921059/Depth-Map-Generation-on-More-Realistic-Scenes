import os
import sys
import numpy as np
from util import cal_avgerr, readPFM

result_dir, gt_dir = sys.argv[1], sys.argv[2]
nums = [str(i) for i in range(10)]
with open('result.txt', 'w') as f:
    for num in nums:
        result = os.path.join(result_dir,'TLD{}.pfm'.format(num))
        gt = os.path.join(gt_dir, 'TLD{}.pfm'.format(num))

        result = readPFM(result).reshape(196608)
        gt = readPFM(gt).reshape(196608)

        f.write(num+'\n')
        f.write(str(cal_avgerr(gt, result))+'\n')
        #f.write('################################################################\n')

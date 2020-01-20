import os
import sys

py_script, data_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
nums = [str(i) for i in range(10)]
os.makedirs(output_dir, exist_ok=True)

for num in nums:
    if os.path.basename(data_dir) == 'Synthetic':
        file_extension = '.png'
    else:
        file_extension = '.bmp'

    left_img = os.path.join(data_dir, 'TL{}'.format(num+file_extension))
    right_img = os.path.join(data_dir, 'TR{}'.format(num+file_extension))
    
    output_pfm = os.path.join(output_dir, 'TLD{}.pfm'.format(num))
    
    print(left_img, right_img, output_pfm)
    
    os.system("python3 %s.py --input-left %s --input-right %s --output %s" %(py_script, left_img, right_img, output_pfm))
    



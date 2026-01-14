import pickle
import numpy as np

file1_path = "/home/huanghao/source/datasets/TWIST2_full/v1_v2_v3_g1/0807_yanjie_walk_002.pkl"
file2_path = "/home/huanghao/source/datasets/gmr_retarget_x/twist2_pico_clean/huanghao2/retargeted_clean/motion_001_part02.pkl"

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data1 = load_pkl(file1_path)
    data2 = load_pkl(file2_path)
    
    list1 = data1.get('link_body_list', [])
    list2 = data2.get('link_body_list', [])
    
    print(f"File 1: {file1_path}")
    print(f"Length of link_body_list: {len(list1)}")
    
    print(f"File 2: {file2_path}")
    print(f"Length of link_body_list: {len(list2)}")
    
    set1 = set(list1)
    set2 = set(list2)
    
    diff1 = set1 - set2
    diff2 = set2 - set1
    
    print("\nDifferences:")
    if diff1:
        print(f"Items in file 1 but not in file 2 ({len(diff1)}):")
        for item in diff1:
            print(f"  - {item}")
    else:
        print("No items in file 1 that are missing from file 2.")
        
    if diff2:
        print(f"Items in file 2 but not in file 1 ({len(diff2)}):")
        for item in diff2:
            print(f"  - {item}")
    else:
        print("No items in file 2 that are missing from file 1.")

except Exception as e:
    print(f"An error occurred: {e}")

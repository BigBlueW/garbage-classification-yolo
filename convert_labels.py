import os
import glob

def main():
    # Original Roboflow/Kaggle classes:
    # 0: BIODEGRADABLE
    # 1: CARDBOARD
    # 2: GLASS
    # 3: METAL
    # 4: PAPER
    # 5: PLASTIC
    
    # Our Target classes:
    # 0: plastic
    # 1: metal
    # 2: paper
    # 3: general_waste
    
    mapping = {
        '5': '0', # PLASTIC -> plastic
        '3': '1', # METAL -> metal
        '1': '2', # CARDBOARD -> paper
        '4': '2', # PAPER -> paper
        '0': '3', # BIODEGRADABLE -> general_waste
        '2': '3', # GLASS -> general_waste
    }

    dataset_path = 'archive/GARBAGE CLASSIFICATION'
    splits = ['train', 'valid', 'test']

    for split in splits:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        print(f"Processing {len(txt_files)} files in {labels_dir}...")
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    old_class = parts[0]
                    if old_class in mapping:
                        parts[0] = mapping[old_class]
                        new_lines.append(' '.join(parts) + '\n')
            
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
                
    print("Label conversion completed successfully!")

if __name__ == '__main__':
    main()

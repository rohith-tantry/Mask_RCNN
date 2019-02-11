import os
import glob
import json
import sys







def insert_json():
     for folder in os.listdir():
        json_files_left = [json_file for json_file in os.listdir(folder) if json_file.endswith('.left.json')]
        for file in json_files_left:
            with open(os.path.join(dataset_dir_path,folder,file),'r') as j_file:
                data = json.loads(j_file.read())
                f_name =[]
                f_name.append(folder)
                f_name.append(file)
                
                data["filename"] = '.'.join(f_name)
                image = file.split('.')
                del image[-1]
                image.append('jpg')
                image_name = '.'.join(image)
                data["image_path"] = os.path.join(dataset_dir_path,folder,image_name)
            with open(os.path.join(dataset_dir_path,folder,file),'w') as f:
                f.write(json.dumps(data, indent = 2))
     

def collect_json():
   
    all_json_files = []
    for folder in os.listdir():
        for file in os.listdir(folder):
            if file.endswith('left.json'):
                all_json_files.append(os.path.join(dataset_dir_path,folder,file))
    return all_json_files
   


def merge_json(input_filenames):
    
    merged = []
    with open('merged_val.json','w') as outfile:
        for file in input_filenames:
            with open(file, 'r') as f:
                data = json.load(f)
                merged.append(data)
        json.dump(merged,outfile,indent =2)




if __name__ == '__main__':
    
    dataset_dir_path = '/home_local/kala_ro/fat/val/'

    os.chdir(dataset_dir_path)

    insert_json()
    input_files = collect_json()
    merge_json(input_files)
    print('done')
    


      
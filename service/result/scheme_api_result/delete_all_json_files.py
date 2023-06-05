import os
import glob

# 注意! 本脚本用于永久删除当前文件夹下所有JSON文件,不可撤销.
# Set the folder path where JSON files are located
folder_path = os.getcwd()

# Get a list of all JSON files in the folder
json_files = glob.glob(os.path.join(folder_path, "*.json"))

# Delete each JSON file in the list
for file in json_files:
    os.remove(file)  

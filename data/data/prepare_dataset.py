import os
from tqdm import tqdm
import json


class DatasetPreparation:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path  # .\\dataset

    def prepare_data(self):
        source_folder_path = os.path.join(self.data_folder_path, "original")

        official_data = []
        i = 0
        for cluster in tqdm(os.listdir(source_folder_path)):
            cluster_folder_path = os.path.join(source_folder_path, f"{cluster}\\original")
            for file_name in os.listdir(cluster_folder_path):
                file_path = os.path.join(cluster_folder_path, file_name)
                file_content = self._read_txt_file(file_path)
                format_content = self._format_content(file_content)

                official_data.append({
                    "id": i,
                    "cluster": cluster[-3:],
                    "text": format_content["Content"],
                    "topic": file_name[:-4],
                    "summary": format_content["Summary"]
                })

                i += 1
        
        self._save_json_file(os.path.join(self.data_folder_path, "official\\official_data.json"), official_data)
        
    def _read_txt_file(self, file_path):
        with open(file_path, encoding="utf-8") as file:
            file_content = file.readlines()
        return file_content
    
    def _format_content(self, file_content):
        format_content = {}
        for i in range(7):
            line_content = file_content[i].split(":")
            format_content[line_content[0].strip()] = line_content[1].strip()

        texts = [text.strip() for text in file_content[8:]]
        
        format_content[file_content[7].strip()[:-1]] = texts

        return format_content
    
    def _save_json_file(self, file_path, data):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_preparation = DatasetPreparation("dataset")
    data_preparation.prepare_data()

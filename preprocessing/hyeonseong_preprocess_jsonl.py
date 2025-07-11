# gdrive_url를 읽어 원천데이터를 불러와 압축해제 후 content만 추출하여 병합
# 7zip download 필요 : 7zip download = https://www.7-zip.org/download.html
# 압축해제 시 2시간 정도 소요, merge_content_download()로 merged_data download 후 read_txt_file()로 txt파일 호출 권장
# 최초 작성일 : 2025-06-02
# 최초 작성자 : 손현성 

import os
import re
import gdown
import tarfile
import subprocess
import json
import shutil

# 따로 cofig 파일로 빼는것이 좋아보임
file_name_and_path = {
    "chokudae.tar": 'https://drive.google.com/file/d/1GCJXcgTEYr2ShmmppH3QUsXOqzvqukTd/view?usp=sharing',
    "pilsu.tar": 'https://drive.google.com/file/d/1TiaGiXVm4P0-wHLBBSkz48EF_GyqDLh2/view?usp=sharing',
    "pro.tar" : 'https://drive.google.com/file/d/1YgigfULJ5qySzFUV7s8St-XPviC7rIR4/view?usp=sharing'
}

part_file_dirs = [
    r"./data/08.전문_의학지식_데이터/3.개방데이터/1.데이터/Training/01.원천데이터",
    r"./data/09.필수의료_의학지식_데이터/3.개방데이터/1.데이터/Training/01.원천데이터",
    r"data/120.초거대AI_사전학습용_헬스케어_질의응답_데이터/3.개방데이터/1.데이터/Training/01.원천데이터"
]
extract_base_dir = r".\data"
seven_zip_path = r"C:\Program Files\7-Zip\7z.exe"
merged_data_dict = {'merge_data.txt': 'https://drive.google.com/file/d/1VbTb9T5nT90J1JJQ2mdX_XaxWfFD3XUq/view?usp=sharing'}

class DataDownLoad:
    """
    gdrive_url에서 원천데이터 다운받아 압축해제
    """
    def __init__(self, file_name_and_path:str=file_name_and_path, extract_base_dir:str = extract_base_dir):
        """
        클래스 초기화 함수

        Input:
            file_name_and_path = {'생성할 파일명' : gdrive_url}
            extract_base_dir = 압축 해제 파일 저장할 경로
        """
        self.file_name_and_path = file_name_and_path  # 꼭 저장해야 함
        self.extract_base_dir = extract_base_dir
        self.file_id = self.extract_file_id(self.file_name_and_path)
        self.file_names = self.file_name_list(self.file_name_and_path)
    
    def extract_file_id(self, path_info:str=file_name_and_path) -> list:
        """
        공유 URL에서 Google Drive 파일 ID 추출

        Input:
            path_info = {'생설할 파일명' : 'gdrive_path'}
        Return:
            result = {'생성할 파일명' : gdrive_id}
        """
        result = {}
        for name, path in path_info.items():
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', path)
            id = match.group(1) if match else None
            result.update({name:id})
        return result

    def file_download(self, id_info=None):
        """
        공유 URL에서 Google Drive 파일 다운로드

        Input:
            id_info : {'파일명': 'file_id'} 형식의 딕셔너리 (None이면 self.file_id 사용)
        """
        id_info = id_info or self.file_id  

        os.makedirs(self.extract_base_dir, exist_ok=True)  # 디렉토리 없으면 생성

        for filename, file_id in id_info.items():
            save_path = os.path.join(self.extract_base_dir, filename)

            if os.path.exists(save_path):
                print(f"[스킵] '{save_path}'은(는) 이미 존재합니다.")
            else:
                download_url = f"https://drive.google.com/uc?id={file_id}"
                print(f"[다운로드] '{save_path}'을(를) 다운로드합니다.")
                try:
                    gdown.download(download_url, save_path, quiet=False)
                    print(f"[완료] '{save_path}' 다운로드 완료.")
                except Exception as e:
                    print(f"[에러] '{save_path}' 다운로드 실패: {e}")

    def file_name_list(self, path_info:str=file_name_and_path) -> list:
        """
        파일명 리스트 반환

        Input:
            path_info = {'생성할 파일명': gdrive path 또는 id }
        Return:
            파일명 리스트
        """
        return [name for name in path_info.keys()]

    def url_file_extractall(self, extract_dir=None):
        """
        공유 URL에서 받은 파일을 생성한 directory에서 압축 풀기

        Input:
            extract_dir = 저장할 경로 (None일 경우 self.extract_base_dir 사용)
        """
        extract_dir = extract_dir or self.extract_base_dir
        os.makedirs(extract_dir, exist_ok=True)
        total = len(self.file_id)
        for index, filename in enumerate(self.file_names, start=1):
            print(f"[{index}/{total}] 압축 해제 시작: {filename}")
            tar_path = filename
            try:
                with tarfile.open(tar_path, 'r:*') as tar:
                    tar.extractall(path=extract_dir)
                    print(f"    → 압축 해제 완료: {tar_path} -> {extract_dir}")
                os.remove(tar_path)
                print(f"    → 압축파일 삭제 완료: {tar_path}")
            except FileNotFoundError:
                print(f"    파일이 존재하지 않음: {tar_path}")
            except Exception as e:
                print(f"    오류 발생: {e}")
        print("모든 압축 해제 작업 완료")

    def zip_part0_file_extractall(self, part_file_dirs:list=part_file_dirs, seven_zip_path:str=seven_zip_path, output_dir_base:str=extract_base_dir):
        """
        7zip 프로그램을 사용하여 .zip.part 파일 압축해제
        7zip download = https://www.7-zip.org/download.html

        Input:
            part_file_dirs = zip_part0 파일 경로 리스트
            seven_zip_path = 7zip.exe 파일 경로
            output_dir_base = 압축해제 파일 저장할 경로
        """ 
        for folder in part_file_dirs:
            # .zip.part0 파일만 선택
            part_files = [f for f in os.listdir(folder) if f.endswith('.zip.part0')]
        
            for part0 in part_files:
                # 기준이 되는 압축 세트 이름 추출
                base_name = part0.replace('.zip.part0', '')
                # 같은 이름의 분할 파일들 찾기
                part_pattern = re.compile(re.escape(base_name) + r'\.zip\.part(\d+)$')
                matching_files = []
        
                # 분할 파일 수집
                for f in os.listdir(folder):
                    m = part_pattern.match(f)
                    if m:
                        part_num = int(m.group(1))
                        matching_files.append((part_num, f))
        
                if not matching_files:
                    continue
        
                # 파일 정렬 및 .zip.0000, .zip.0001 리네이밍 (복사본 생성)
                matching_files.sort()
                temp_dir = os.path.join(folder, f"{base_name}_temp_parts")
                os.makedirs(temp_dir, exist_ok=True)
        
                new_filenames = []
                for idx, (_, orig_name) in enumerate(matching_files):
                    new_name = f"{base_name}.zip.{idx:04d}"
                    shutil.copyfile(os.path.join(folder, orig_name), os.path.join(temp_dir, new_name))
                    new_filenames.append(new_name)
        
                # 압축 해제
                first_file = os.path.join(temp_dir, new_filenames[0])
                output_dir = os.path.join(output_dir_base, base_name)
                os.makedirs(output_dir, exist_ok=True)
        
                print(f"압축 해제 중: {first_file} -> {output_dir}")
                subprocess.run([
                    seven_zip_path,
                    "x",
                    first_file,
                    f"-o{output_dir}"
                ])
        
                # 임시 폴더 삭제
                shutil.rmtree(temp_dir)
    
    def load_txt_file(self, file_name='merged_data.txt', split_word='\n\n'):
        """
        txt파일 불러오기

        Input:
            file_name = 불어올 txt파일명
            split_word = content 분류 기준
        Return:
            contents = 불러온 data의 content list
        """
        output_txt_path = os.path.join(self.extract_base_dir, file_name)  
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        contents = [block.strip() for block in raw_text.split(split_word) if block.strip()]
        print(f"총 {len(contents)}개의 블록이 분리되었습니다.")
        return contents
    
    def merge_content_download(self, path_info:str=merged_data_dict):
        """
        원천데이터에서 content만 추출한 데이터 다운로드드
        """
        id_info = self.extract_file_id(path_info)
        self.file_download(id_info)

class DataPreprocessing:
    """
    필요한 data에서 content만 추출하여 병합
    """
    def __init__(self, extract_base_dir='./data'):
        """
        클래스 초기화 함수
    
        Input:
            extract_base_dir = 압축 해제 파일 저장할 경로
        """
        self.extract_base_dir = extract_base_dir

    def get_merge_data(self, search_directory_name='TS_국문', file_name='merged_data.txt', data_key="content"):
        """
        TS_국문 디렉토리 안에 있는 데이터 병합

        Input:
            search_directory_name = 디렉토리명에서 찾을 단어
            file_name = 생성할 병합 데이터 파일명 
            data_key = 추출할 value값의 key 값
        Return:
            all_contents = 병합한 content list
        """
        output_txt_path = os.path.join(self.extract_base_dir, file_name)
        all_contents = []
        
        for folder_name in os.listdir(self.extract_base_dir):
            folder_path = os.path.join(self.extract_base_dir, folder_name)

            if os.path.isdir(folder_path) and search_directory_name in folder_name:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.json'):
                        json_path = os.path.join(folder_path, file_name)
                        try:
                            with open(json_path, 'r', encoding='utf-8-sig') as f:
                                data = json.load(f)
                                if data_key in data:
                                    content = data[data_key]
                                elif "data" in data and data_key in data["data"]:
                                    content = data["data"][data_key]
                                else:
                                    content = None

                                if content:
                                    all_contents.append(content)
                        except Exception as e:
                            print(f"오류: {json_path} → {type(e).__name__}: {e}")

        # 파일 저장
        with open(output_txt_path, 'w', encoding='utf-8') as out_file:
            for content in all_contents:
                out_file.write(content.strip() + '\n\n')

        print(f"완료: {len(all_contents)}개의 content가 '{output_txt_path}'에 저장되었습니다.")
        return all_contents

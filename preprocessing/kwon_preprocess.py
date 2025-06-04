from pathlib import Path
import json
import chardet
from tqdm import tqdm

def load_all_jsons_from_path(path, max_files=None):
    """
    지정된 경로에서 모든 JSON 파일을 재귀적으로 읽어 리스트로 반환.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 경로가 존재하지 않음")

    json_files = list(path.rglob("*.json"))
    if max_files is not None:
        json_files = json_files[:max_files]

    all_data = []
    for json_file in tqdm(json_files, desc="파일 로딩 중"):
        try:
            raw_bytes = json_file.read_bytes()
            encoding = chardet.detect(raw_bytes)["encoding"] or "utf-8"
            data = json.loads(raw_bytes.decode(encoding))
            all_data.append(data)
        except Exception as e:
            print(f"[오류] {json_file.name} 파일 로드 실패: {e}")
    return all_data

# 2. 카테고리별 샘플 데이터 준비 (전처리 포함)
def prepare_category_db(data_list, category):
    processed = []
    for item in data_list:
        if isinstance(item, dict):
            if 'content' in item:
                text = item['content']
            elif 'question' in item:
                text = item['question']
            elif 'answer' in item:
                ans = item['answer']
                text = ' '.join([ans.get(k, '') for k in ['intro', 'body', 'conclusion']])
            else:
                text = str(item)
        else:
            text = str(item)
        processed.append({'content': text.strip(), 'category': category})
    return processed

from pathlib import Path
import os
import random
import json
import re

def get_json_files(folder: Path):
    return [p for p in folder.glob('*.json') if p.is_file()]

def load_question_text(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("question", "").strip()

def load_answer_text(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    answer = data.get("answer", {})
    return answer.get("conclusion", "").strip()

def find_final_subfolders(root_path: Path):
    """ root_path 아래 최하위 폴더(안에 폴더가 없는) 리스트 반환 """
    final_folders = []
    for dirpath, dirnames, _ in os.walk(root_path):
        if not dirnames:  # 더 이상 하위 폴더가 없으면면
            final_folders.append(Path(dirpath))
    return final_folders

def create_txt_from_all_subfolders(base_path, categories, sample_size=3, output_file='combined.txt'):
    base_path = Path(base_path)
    question_root = base_path / '1.질문'
    answer_root = base_path / '2.답변'

    combined_lines = []

    for category in categories:
        q_category_path = question_root / category
        a_category_path = answer_root / category

        q_final_folders = find_final_subfolders(q_category_path)

        for q_folder in q_final_folders:
            # 답변 쪽 동일 경로 찾기
            rel_path = q_folder.relative_to(question_root)
            a_folder = answer_root / rel_path

            if not a_folder.exists():
                # 답변 폴더가 없으면 건너뜀
                continue

            q_files = get_json_files(q_folder)
            a_files = get_json_files(a_folder)

            if len(q_files) < sample_size or len(a_files) < sample_size:
                # 질문이나 답변이 sample_size 미만이면 건너뜀
                continue

            sampled_q = random.sample(q_files, sample_size)
            sampled_a = random.sample(a_files, sample_size)

            for qf, af in zip(sampled_q, sampled_a):
                try:
                    q_text = load_question_text(qf)
                    a_text = load_answer_text(af)
                    combined_lines.append(f"{q_text} {a_text}")
                except Exception as e:
                    print(f"오류 발생 : {qf}, {af} -> {e}")

    if combined_lines:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_lines))
        print(f"텍스트 파일 생성 완료 : {output_file}")
    else:
        print("질문-답변 쌍을 찾지 못했습니다.")

# 한국어 불용어 예시 (필요하면 수정/추가 가능)
STOPWORDS = {
    "그", "이", "저", "것", "수", "등", "들", "및", "또한", "그리고", "하지만", "그러나",
    "때문", "때문에", "때", "더", "더욱", "더군다나", "더라도", "뿐만 아니라", "뿐이다",
    "즉", "예를 들어", "즉시", "아주", "같은", "약간", "약간의", "약간씩", "각", "각각",
    "어떤", "어느", "어느 정도", "어느 정도로", "어느 정도의", "어느 정도까지", "어느 정도까지의"
}

def preprocess_text(text: str, stopwords=STOPWORDS) -> str:
    # 1) 알파벳-단어, 한글, 숫자, 띄어쓰기만 유지
    #    영어 단어 또는 영어-영어 단어를 포함하도록 패턴 허용
    text = re.sub(r'[^가-힣a-zA-Z0-9\s\-]', ' ', text)

    # 2) 하이픈이 단어 사이에 있는 경우만 유지, 앞뒤 공백 제거
    text = re.sub(r'\s*-\s*', '-', text)  # 공백 있는 하이픈을 붙여줌
    text = re.sub(r'(^|\s)-|-(\s|$)', ' ', text)  # 단독 하이픈 제거

    # 3) 여러 공백은 하나 공백으로
    text = re.sub(r'\s+', ' ', text).strip()

    # 4) 불용어 제거
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stopwords]
    return ' '.join(filtered_words)


def preprocess_file(input_path: str, output_path: str, stopwords=STOPWORDS):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    preprocessed_lines = [preprocess_text(line, stopwords) for line in lines]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(preprocessed_lines))
    print(f"전처리 완료 : {output_path}")

# 사용 예시
if __name__ == "__main__":
    base_path = "./초거대 AI 헬스케어 질의응답 데이터/TS"   # 데이터가 있는 최상위 폴더 경로
    categories = ["감염성질환","근골격질환","뇌신경정신질환","눈질환","소아청소년질환","소화기질환","순환기질환","신장비뇨기질환","여성질환","유방내분비질환","유전질환","응급질환","호흡기질환"]
    output_file = "QA_random_pair_part1.txt"
    create_txt_from_all_subfolders(base_path, categories, sample_size=3, output_file=output_file)

    input_file = "QA_random_pair_part1.txt"
    output_file = "QA_random_pair_part1_preprocessed.txt"
    preprocess_file(input_file, output_file)
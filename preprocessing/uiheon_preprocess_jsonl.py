import json

def convert_qna_to_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in lines:
            if '?' not in line:
                print(f"물음표(?) 없는 라인 건너뜀: {line}")
                continue

            question, answer = line.split('?', 1)
            question += '?'  # 물음표를 질문 끝에 다시 붙이기

            obj = {
                "messages": [
                    {"role": "user", "content": question.strip()},
                    {"role": "assistant", "content": answer.strip()}
                ]
            }

            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"변환 완료, {output_path}에 저장했습니다.")

# 사용 예시
convert_qna_to_jsonl("./QA_random_pair_part1.txt", "./finetune_data_QApart1.jsonl")
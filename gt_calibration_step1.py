import os
import re
import csv
import unicodedata
from difflib import SequenceMatcher


# ==================================================
# 1. 路径配置
# ==================================================
WHISPER_DIR = "/data2/fanl/M2Voice/dataset/dt4/txt"
OUTPUT_DIR = "/data2/fanl/M2Voice/dataset/dt4/txt_calibrated"
GT_FILE = "/data2/fanl/M2Voice/dataset/dt4/gt.txt"
REPORT_FILE = "/data2/fanl/M2Voice/dataset/dt4/calibration_report.csv"


# 最佳候选的最低相似度
MIN_SCORE = 0.45

# 最佳候选与第二候选的最低相似度差
MIN_MARGIN = 0.10


# ==================================================
# 2. 文本标准化
# ==================================================
def normalize_text(text):
    """
    对文本进行标准化：
    1. 统一全角和半角
    2. 转换为小写
    3. 统一部分数字表达
    4. 去除标点和多余空格
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    replacements = {
        "7:30": "seven thirty",
        "7 30": "seven thirty",
        "seven-thirty": "seven thirty",

        "5:00": "five",
        "5 pm": "five pm",
        "five p.m.": "five pm",

        "30%": "thirty percent",
        "30 percent": "thirty percent",
        "30 seconds": "thirty seconds",

        "$50": "fifty us dollars",
        "50 us dollars": "fifty us dollars",
        "50 dollars": "fifty us dollars",

        "p.m.": "pm",
        "a.m.": "am",

        "u.s.": "us",
        "u.s": "us",

        "euro": "euros",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    # 去除标点，只保留字母、数字、下划线和空格
    text = re.sub(r"[^\w\s]", " ", text)

    # 合并连续空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==================================================
# 3. 计算文本相似度
# ==================================================
def similarity(text1, text2):
    """
    结合字符级和单词级相似度。

    返回值范围：
        0.0：完全不同
        1.0：完全相同
    """
    normalized_text1 = normalize_text(text1)
    normalized_text2 = normalize_text(text2)

    if not normalized_text1 or not normalized_text2:
        return 0.0

    # 字符级相似度
    char_score = SequenceMatcher(
        None,
        normalized_text1,
        normalized_text2
    ).ratio()

    # 单词级相似度
    words1 = normalized_text1.split()
    words2 = normalized_text2.split()

    word_score = SequenceMatcher(
        None,
        words1,
        words2
    ).ratio()

    # 单词级相似度权重更高
    final_score = 0.4 * char_score + 0.6 * word_score

    return final_score


# ==================================================
# 4. 读取GT标准句
# ==================================================
def load_gt_sentences(gt_file):
    if not os.path.isfile(gt_file):
        raise FileNotFoundError(f"GT文件不存在：{gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(sentences) != 20:
        raise ValueError(
            f"{gt_file} 应包含20条非空标准句，"
            f"当前读取到 {len(sentences)} 条"
        )

    return sentences


# ==================================================
# 5. 校准单个Whisper文本
# ==================================================
def calibrate_text(filename, whisper_text, gt_sentences):
    """
    文件名中的s1～s10分别对应两个候选GT：

        s1  -> GT 1、GT 11
        s2  -> GT 2、GT 12
        ...
        s10 -> GT 10、GT 20
    """
    match = re.search(
        r"_s(\d+)\.txt$",
        filename,
        re.IGNORECASE
    )

    if not match:
        return {
            "success": False,
            "reason": "bad_filename",
            "suffix_id": None,
            "best_sid": None,
            "best_score": 0.0,
            "second_sid": None,
            "second_score": 0.0,
            "margin": 0.0,
            "best_gt": "",
            "second_gt": "",
            "output_text": whisper_text,
            "status": "manual_review",
        }

    suffix_id = int(match.group(1))

    if suffix_id < 1 or suffix_id > 10:
        return {
            "success": False,
            "reason": "invalid_suffix",
            "suffix_id": suffix_id,
            "best_sid": None,
            "best_score": 0.0,
            "second_sid": None,
            "second_score": 0.0,
            "margin": 0.0,
            "best_gt": "",
            "second_gt": "",
            "output_text": whisper_text,
            "status": "manual_review",
        }

    # 每个后缀只对应两个可能的标准句
    candidate_sids = [
        suffix_id,
        suffix_id + 10,
    ]

    candidates = []

    for sid in candidate_sids:
        gt_text = gt_sentences[sid - 1]
        score = similarity(whisper_text, gt_text)

        candidates.append({
            "sid": sid,
            "text": gt_text,
            "score": score,
        })

    # 相似度从高到低排序
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    best_candidate = candidates[0]
    second_candidate = candidates[1]

    best_score = best_candidate["score"]
    second_score = second_candidate["score"]
    margin = best_score - second_score

    # 只有匹配足够可靠时才校准
    if best_score >= MIN_SCORE and margin >= MIN_MARGIN:
        output_text = best_candidate["text"]
        status = "calibrated"
        success = True
        reason = "accepted"
    else:
        # 不确定时保留Whisper原始文本
        output_text = whisper_text
        status = "manual_review"
        success = False
        reason = "low_confidence"

    return {
        "success": success,
        "reason": reason,
        "suffix_id": suffix_id,
        "best_sid": best_candidate["sid"],
        "best_score": best_score,
        "second_sid": second_candidate["sid"],
        "second_score": second_score,
        "margin": margin,
        "best_gt": best_candidate["text"],
        "second_gt": second_candidate["text"],
        "output_text": output_text,
        "status": status,
    }


# ==================================================
# 6. 主程序
# ==================================================
def main():
    if not os.path.isdir(WHISPER_DIR):
        raise FileNotFoundError(
            f"Whisper文本目录不存在：{WHISPER_DIR}"
        )

    gt_sentences = load_gt_sentences(GT_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_rows = []

    processed_count = 0
    calibrated_count = 0
    review_count = 0
    skipped_count = 0

    filenames = sorted(os.listdir(WHISPER_DIR))

    txt_files = [
        filename
        for filename in filenames
        if filename.lower().endswith(".txt")
    ]

    print(f"Whisper directory : {WHISPER_DIR}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"GT file           : {GT_FILE}")
    print(f"TXT files         : {len(txt_files)}")
    print()

    for index, filename in enumerate(txt_files, start=1):
        input_path = os.path.join(
            WHISPER_DIR,
            filename
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        with open(input_path, "r", encoding="utf-8") as f:
            whisper_text = f.read().strip()

        processed_count += 1

        result = calibrate_text(
            filename,
            whisper_text,
            gt_sentences
        )

        if result["reason"] in {
            "bad_filename",
            "invalid_suffix",
        }:
            skipped_count += 1

        if result["status"] == "calibrated":
            calibrated_count += 1
        else:
            review_count += 1

        # 写入校准后的文本
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["output_text"] + "\n")

        report_rows.append({
            "filename": filename,
            "suffix_id": result["suffix_id"],
            "best_gt_id": result["best_sid"],
            "second_gt_id": result["second_sid"],
            "best_similarity": round(
                result["best_score"],
                4
            ),
            "second_similarity": round(
                result["second_score"],
                4
            ),
            "margin": round(
                result["margin"],
                4
            ),
            "status": result["status"],
            "reason": result["reason"],
            "whisper_text": whisper_text,
            "best_gt": result["best_gt"],
            "second_gt": result["second_gt"],
            "output_text": result["output_text"],
        })

        print("=" * 80)
        print(f"[{index}/{len(txt_files)}]")
        print(f"File              : {filename}")
        print(f"Whisper           : {whisper_text}")
        print(f"File suffix       : s{result['suffix_id']}")
        print(
            f"Best GT           : "
            f"GT {result['best_sid']} - {result['best_gt']}"
        )
        print(
            f"Second GT         : "
            f"GT {result['second_sid']} - {result['second_gt']}"
        )
        print(
            f"Best similarity   : "
            f"{result['best_score']:.4f}"
        )
        print(
            f"Second similarity : "
            f"{result['second_score']:.4f}"
        )
        print(f"Margin            : {result['margin']:.4f}")
        print(f"Status            : {result['status']}")
        print(f"Reason            : {result['reason']}")
        print(f"Output            : {result['output_text']}")

    # ==================================================
    # 7. 保存CSV报告
    # ==================================================
    fieldnames = [
        "filename",
        "suffix_id",
        "best_gt_id",
        "second_gt_id",
        "best_similarity",
        "second_similarity",
        "margin",
        "status",
        "reason",
        "whisper_text",
        "best_gt",
        "second_gt",
        "output_text",
    ]

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 80)
    print("Calibration finished.")
    print(f"Processed     : {processed_count}")
    print(f"Calibrated    : {calibrated_count}")
    print(f"Manual review : {review_count}")
    print(f"Skipped       : {skipped_count}")
    print(f"Output        : {OUTPUT_DIR}")
    print(f"Report        : {REPORT_FILE}")


if __name__ == "__main__":
    main()
"""把 data/ 目录下所有 txt 批量入库到 Chroma。"""
import glob
import os

from knowledge_base import KnowledgeBaseService


def main():
    service = KnowledgeBaseService()

    files = sorted(glob.glob("data/*.txt"))
    print(f"待入库文件数：{len(files)}")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        filename = os.path.basename(path)
        result = service.upload_by_str(text, filename)
        print(f"{result}  {filename}")


if __name__ == "__main__":
    main()

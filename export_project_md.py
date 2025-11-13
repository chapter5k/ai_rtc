#!/usr/bin/env python
# export_project_md.py
"""
프로젝트 루트에서 실행하면,
- 현재 디렉토리 트리
- 각 .py 파일의 전체 코드

를 하나의 Markdown 파일(project_snapshot.md)로 내보내는 스크립트.
"""

import os
from datetime import datetime

# 무시할 디렉토리들 (필요하면 추가/삭제)
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "outputs",
    "venv",
    ".idea",
    "old",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
}

# 무시할 파일명들 (원하면 여기에 자신을 넣어도 됨: "export_project_md.py")
EXCLUDE_FILES = {
    "export_project_md.py",
    "스크립트 실행.docx",
}


def is_ignored_dir(dirname: str) -> bool:
    return dirname in EXCLUDE_DIRS


def is_ignored_file(filename: str) -> bool:
    if not filename.endswith(".py"):
        return True  # .py가 아니면 스냅샷 대상에서 제외
    if filename in EXCLUDE_FILES:
        return True
    return False


def collect_py_files(root: str):
    """root 아래의 모든 .py 파일 경로(상대경로) 리스트를 수집."""
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 무시할 디렉토리 제거
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(d)]

        for fn in filenames:
            if is_ignored_file(fn):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, root)
            py_files.append(rel_path)

    py_files.sort()
    return py_files


def build_tree(root: str) -> str:
    """
    간단한 텍스트 디렉토리 트리 문자열 생성.
    (ls -R / tree 느낌)
    """
    lines = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 무시할 디렉토리 제거
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(d)]

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = "."

        lines.append(rel_dir + "/")

        # 디렉토리 안 파일들
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue  # 숨김 파일은 대충 무시
            lines.append(f"  {fn}")

        lines.append("")  # 디렉토리 간 빈 줄

    return "\n".join(lines).rstrip()  # 마지막 공백 제거


def read_file_content(root: str, rel_path: str) -> str:
    """상대 경로 기준 파일 내용을 UTF-8로 읽는다."""
    full_path = os.path.join(root, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_markdown(root: str, output_path: str):
    """project_snapshot.md 를 생성."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tree_str = build_tree(root)
    py_files = collect_py_files(root)

    lines = []
    lines.append("# Project Snapshot")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Root directory: `{root}`")
    lines.append("")
    lines.append("## Directory Tree")
    lines.append("")
    lines.append("```text")
    lines.append(tree_str)
    lines.append("```")
    lines.append("")

    lines.append("## Python Files")
    lines.append("")

    if not py_files:
        lines.append("_No Python files found (after filters)._")
    else:
        for rel_path in py_files:
            lines.append(f"### `{rel_path}`")
            lines.append("")
            lines.append("```python")
            try:
                content = read_file_content(root, rel_path)
                lines.append(content)
            except Exception as e:
                lines.append(f"# [ERROR] 파일을 읽는 중 문제가 발생했습니다: {e}")
            lines.append("```")
            lines.append("")

    md_text = "\n".join(lines)

    # 출력 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"[완료] 프로젝트 스냅샷 저장: {output_path}")


def main():
    # 이 스크립트가 있는 위치를 프로젝트 루트로 간주
    root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root, "project_snapshot.md")

    generate_markdown(root, output_path)


if __name__ == "__main__":
    main()

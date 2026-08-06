from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

REPOSITORY_ROOT = Path(os.environ.get("REPOSITORY_ROOT_PHP", ".")).expanduser().resolve()
MAX_OUTPUT_CHARS = int(os.environ.get("GIT_TOOL_MAX_OUTPUT_CHARS", "100000"))
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("GIT_TOOL_TIMEOUT_SECONDS", "30"))


class GitToolError(RuntimeError):
    """Raised when a read-only repository operation fails."""


def _ensure_repository() -> None:
    if not REPOSITORY_ROOT.exists():
        raise GitToolError(f"Repository path does not exist: {REPOSITORY_ROOT}")

    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    if result.returncode != 0 or result.stdout.strip() != "true":
        raise GitToolError(f"Not a Git working tree: {REPOSITORY_ROOT}")


def _truncate_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return output[:MAX_OUTPUT_CHARS] + "\n\n[Output truncated.]"


def _run_readonly_command(
    command: list[str],
    *,
    cwd: Path = REPOSITORY_ROOT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allowed_return_codes: tuple[int, ...] = (0,),
) -> str:
    if not command:
        raise GitToolError("Command cannot be empty.")

    resolved_cwd = cwd.expanduser().resolve()
    try:
        resolved_cwd.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise GitToolError(f"Path outside repository: {resolved_cwd}") from exc

    if command[0] not in {"git", "rg"}:
        raise GitToolError(f"Command is not allowed: {command[0]}")

    env = {
        **os.environ,
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }

    try:
        result = subprocess.run(
            command,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise GitToolError(f"Required command not installed: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitToolError(f"Command timed out after {timeout_seconds} seconds.") from exc

    output = "\n".join(
        section for section in (result.stdout.strip(), result.stderr.strip()) if section
    )

    if result.returncode not in allowed_return_codes:
        raise GitToolError(
            "Read-only command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {result.returncode}\n"
            f"Output:\n{output or '[no output]'}"
        )

    return _truncate_output(output)


def _resolve_repository_path(file_path: str) -> Path:
    if not file_path or not file_path.strip():
        raise GitToolError("File path cannot be empty.")

    requested_path = (REPOSITORY_ROOT / file_path).expanduser().resolve()
    try:
        requested_path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise GitToolError(f"Requested path is outside repository: {file_path}") from exc
    return requested_path


def _safe_git_ref(git_ref: str) -> str:
    if not git_ref or not git_ref.strip():
        raise GitToolError("Git reference cannot be empty.")

    resolved = _run_readonly_command(
        ["git", "rev-parse", "--verify", f"{git_ref}^{{commit}}"]
    )
    return resolved.splitlines()[0].strip()


def _parse_name_status(output: str) -> list[dict[str, Optional[str]]]:
    files: list[dict[str, Optional[str]]] = []

    for line in output.splitlines():
        if not line.strip():
            continue

        fields = line.split("\t")
        status = fields[0]

        if len(fields) == 2:
            files.append({"status": status, "path": fields[1], "previous_path": None})
        elif len(fields) >= 3:
            files.append(
                {"status": status, "path": fields[-1], "previous_path": fields[-2]}
            )

    return files



def get_repository_status() -> dict[str, Any]:
    """Return root, current branch, HEAD commit, and working-tree status."""
    _ensure_repository()

    branch = _run_readonly_command(["git", "branch", "--show-current"])
    status = _run_readonly_command(["git", "status", "--short", "--branch"])
    head_commit = _run_readonly_command(["git", "rev-parse", "HEAD"])
    repository_root = _run_readonly_command(["git", "rev-parse", "--show-toplevel"])

    changed_lines = [
        line for line in status.splitlines() if line and not line.startswith("##")
    ]

    return {
        "repository_root": repository_root,
        "current_branch": branch or None,
        "head_commit": head_commit,
        "status": status,
        "is_clean": len(changed_lines) == 0,
    }



def list_git_branches(
    include_remote: bool = True,
    contains_text: str = "",
) -> dict[str, Any]:
    """List local and optionally remote branches, with optional filtering."""
    _ensure_repository()

    command = ["git", "branch", "--no-color", "--format=%(refname:short)"]
    if include_remote:
        command.append("--all")

    output = _run_readonly_command(command)
    branches = [line.strip() for line in output.splitlines() if line.strip()]

    if contains_text:
        value = contains_text.casefold()
        branches = [branch for branch in branches if value in branch.casefold()]

    return {"count": len(branches), "branches": branches}


def search_git_history(
    query: str,
    max_results: int = 50,
    all_branches: bool = True,
) -> dict[str, Any]:
    """Search commit messages, for example for `TRAN-4072`."""
    _ensure_repository()

    if not query or not query.strip():
        raise GitToolError("Search query cannot be empty.")

    max_results = max(1, min(max_results, 200))
    command = [
        "git",
        "log",
        f"--max-count={max_results}",
        "--date=iso-strict",
        "--pretty=format:%H%x09%ad%x09%an%x09%s",
        "--regexp-ignore-case",
        f"--grep={query}",
    ]

    if all_branches:
        command.insert(2, "--all")

    output = _run_readonly_command(command)
    commits: list[dict[str, str]] = []

    for line in output.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            commit_hash, date, author, subject = parts
            commits.append(
                {
                    "commit": commit_hash,
                    "date": date,
                    "author": author,
                    "subject": subject,
                }
            )

    return {"query": query, "count": len(commits), "commits": commits}



def read_git_commit(
    commit_ref: str,
    include_patch: bool = False,
) -> dict[str, Any]:
    """Read commit metadata, changed files, and optionally its patch."""
    _ensure_repository()
    resolved_commit = _safe_git_ref(commit_ref)

    metadata = _run_readonly_command(
        [
            "git",
            "show",
            "--no-patch",
            "--date=iso-strict",
            "--format=fuller",
            resolved_commit,
        ]
    )

    changed_files_output = _run_readonly_command(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--find-renames",
            "-r",
            resolved_commit,
        ]
    )

    response: dict[str, Any] = {
        "requested_ref": commit_ref,
        "resolved_commit": resolved_commit,
        "metadata": metadata,
        "changed_files": _parse_name_status(changed_files_output),
    }

    if include_patch:
        response["patch"] = _run_readonly_command(
            [
                "git",
                "show",
                "--format=fuller",
                "--find-renames",
                "--find-copies",
                resolved_commit,
            ],
            timeout_seconds=60,
        )

    return response



def list_changed_files(base_ref: str, head_ref: str) -> dict[str, Any]:
    """List files changed between two refs using `base...head`."""
    _ensure_repository()
    _safe_git_ref(base_ref)
    _safe_git_ref(head_ref)

    comparison = f"{base_ref}...{head_ref}"
    output = _run_readonly_command(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            comparison,
        ]
    )

    files = _parse_name_status(output)
    return {
        "base_ref": base_ref,
        "head_ref": head_ref,
        "comparison": comparison,
        "count": len(files),
        "files": files,
    }



def read_git_diff(
    base_ref: str,
    head_ref: str,
    file_path: str = "",
    context_lines: int = 5,
) -> dict[str, Any]:
    """Read patch, statistics, and changed files between two refs."""
    _ensure_repository()
    _safe_git_ref(base_ref)
    _safe_git_ref(head_ref)

    context_lines = max(0, min(context_lines, 50))
    comparison = f"{base_ref}...{head_ref}"

    patch_command = [
        "git",
        "diff",
        "--no-ext-diff",
        "--find-renames",
        "--find-copies",
        f"--unified={context_lines}",
        comparison,
    ]

    if file_path:
        safe_path = _resolve_repository_path(file_path)
        relative_path = safe_path.relative_to(REPOSITORY_ROOT).as_posix()
        patch_command.extend(["--", relative_path])

    patch = _run_readonly_command(patch_command, timeout_seconds=60)
    changed_files_output = _run_readonly_command(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            comparison,
        ]
    )
    statistics = _run_readonly_command(["git", "diff", "--stat", comparison])

    return {
        "base_ref": base_ref,
        "head_ref": head_ref,
        "comparison": comparison,
        "file_path": file_path or None,
        "statistics": statistics,
        "changed_files": _parse_name_status(changed_files_output),
        "patch": patch,
    }



def read_file_at_git_ref(
    git_ref: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 0,
) -> dict[str, Any]:
    """Read a text file at a branch, tag, or commit without checkout."""
    _ensure_repository()
    _safe_git_ref(git_ref)

    safe_path = _resolve_repository_path(file_path)
    relative_path = safe_path.relative_to(REPOSITORY_ROOT).as_posix()
    content = _run_readonly_command(["git", "show", f"{git_ref}:{relative_path}"])

    lines = content.splitlines()
    first_line = max(start_line, 1)
    last_line = len(lines) if end_line <= 0 else max(first_line, min(end_line, len(lines)))
    selected_lines = lines[first_line - 1:last_line]

    numbered_content = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(selected_lines, start=first_line)
    )

    return {
        "git_ref": git_ref,
        "path": relative_path,
        "start_line": first_line,
        "end_line": last_line,
        "total_lines": len(lines),
        "content": numbered_content,
    }



def read_repository_file(
    file_path: str,
    start_line: int = 1,
    end_line: int = 0,
) -> dict[str, Any]:
    """Read a UTF-8 text file from the current working tree."""
    _ensure_repository()
    resolved_path = _resolve_repository_path(file_path)

    if not resolved_path.exists():
        raise GitToolError(f"File does not exist: {file_path}")
    if not resolved_path.is_file():
        raise GitToolError(f"Path is not a file: {file_path}")

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise GitToolError(f"File is not UTF-8 text: {file_path}") from exc

    lines = content.splitlines()
    first_line = max(start_line, 1)
    last_line = len(lines) if end_line <= 0 else max(first_line, min(end_line, len(lines)))
    selected_lines = lines[first_line - 1:last_line]

    numbered_content = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(selected_lines, start=first_line)
    )

    return {
        "path": resolved_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "start_line": first_line,
        "end_line": last_line,
        "total_lines": len(lines),
        "content": numbered_content,
    }


def search_code(
    query: str,
    path: str = "",
    file_glob: str = "",
    max_results: int = 200,
    fixed_string: bool = True,
) -> dict[str, Any]:
    """Search repository text using ripgrep without executing project code."""
    _ensure_repository()

    if not query or not query.strip():
        raise GitToolError("Search query cannot be empty.")

    max_results = max(1, min(max_results, 1000))
    command = ["rg", "--line-number", "--column", "--no-heading", "--color", "never"]

    if fixed_string:
        command.append("--fixed-strings")
    if file_glob:
        command.extend(["--glob", file_glob])

    command.append(query)

    if path:
        safe_path = _resolve_repository_path(path)
        command.append(safe_path.relative_to(REPOSITORY_ROOT).as_posix())
    else:
        command.append(".")

    output = _run_readonly_command(command, allowed_return_codes=(0, 1))

    if not output:
        return {"query": query, "count": 0, "matches": []}

    matches: list[dict[str, Any]] = []

    for line in output.splitlines():
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue

        file_name, line_number, column_number, text = parts
        try:
            parsed_line = int(line_number)
            parsed_column = int(column_number)
        except ValueError:
            continue

        matches.append(
            {
                "path": file_name,
                "line": parsed_line,
                "column": parsed_column,
                "text": text,
            }
        )

        if len(matches) >= max_results:
            break

    return {"query": query, "count": len(matches), "matches": matches}


def find_changed_file_references(
    base_ref: str,
    head_ref: str,
    max_results_per_name: int = 50,
) -> dict[str, Any]:
    """Find static text references to names of changed files and modules."""
    changed = list_changed_files.invoke({"base_ref": base_ref, "head_ref": head_ref})
    results: list[dict[str, Any]] = []

    for changed_file in changed["files"]:
        path = Path(str(changed_file["path"]))
        candidate_names = {path.name, path.stem}
        reference_results: list[dict[str, Any]] = []

        for candidate in sorted(candidate_names):
            if len(candidate) < 3:
                continue

            search_result = search_code.invoke(
                {
                    "query": candidate,
                    "max_results": max_results_per_name,
                    "fixed_string": True,
                }
            )

            reference_results.append(
                {"query": candidate, "matches": search_result["matches"]}
            )

        results.append(
            {
                "changed_file": changed_file["path"],
                "references": reference_results,
            }
        )

    return {"base_ref": base_ref, "head_ref": head_ref, "results": results}


GIT_TOOLS = [
    get_repository_status,
    list_git_branches,
    search_git_history,
    read_git_commit,
    list_changed_files,
    read_git_diff,
    read_file_at_git_ref,
    read_repository_file,
    search_code,
    find_changed_file_references,
]


if __name__ == "__main__":
    print(get_repository_status())
    print(list_git_branches())
    print(search_git_history("add support for"))
#     print(get_repository_status.invoke({}))
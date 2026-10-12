"""Check that only comments and docstrings changed: each file's syntax tree, docstrings removed, against git's.

    python testing/code_unchanged.py [base revision, default HEAD] [files, default every changed .py file]
"""
import ast
import subprocess
import sys


def code_tree(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def main(base='HEAD', files=None):
    if not files:
        files = subprocess.run(['git', 'diff', '--name-only', base, '--', '*.py'], capture_output=True, text=True,
                               check=True).stdout.split()
    changed = []
    for path in files:
        before = subprocess.run(['git', 'show', f'{base}:{path}'], capture_output=True, text=True, encoding='utf-8')
        if before.returncode != 0:
            continue
        with open(path, encoding='utf-8') as file:
            if code_tree(before.stdout) != code_tree(file.read()):
                changed.append(path)
    for path in changed:
        print(f'code changed: {path}')
    print(f'{len(files)} files checked, {len(changed)} with code changes')
    return 1 if changed else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:2], files=sys.argv[2:]))

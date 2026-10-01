#!/usr/bin/env python3
"""
studio.py — AgentVideo 更新工具（只拉取 .py 文件）

用法:
    python studio.py                 检查更新
    python studio.py check           同上
    python studio.py diff            显示哪些 .py 变了
    python studio.py pull            只下载变更的 .py 到 _update/
    python studio.py info            显示版本信息
    python studio.py verify          校验本地文件完整性
    python studio.py scan            列出会跟踪的 .py 文件
    python studio.py --url <URL>     自定义 version.json 地址

配置文件 (studio.conf.json，可选):
    {
      "update_url": "https://justforhavefun.github.io/AgentVideo/version.json",
      "local_version": "1.0.0",
      "suffixes": [".py"]
    }
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_UPDATE_URL = (
    'https://justforhavefun.github.io/AgentVideo/version.json'
)
CONF_NAME = 'studio.conf.json'
HASH_ALGO = 'sha256'

# 默认跟踪规则
DEFAULT_SUFFIXES = ['.py']

IGNORE_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', 'node_modules',
    'tmp', 'backups', 'vcompare', 'contrast', 'compare_out',
    '.vscode', '.idea', '.pytest_cache', '.mypy_cache',
    'moss_tts', 'build', 'dist', '.venv', 'venv', 'env',
    '_update',
}

IGNORE_PATTERNS = [
    '*.bak', '*.bak_*', '*.orig', '*.pyc', '*.pyo',
    '*.log',
]

IGNORE_NAMES = {
    'version.json',
    CONF_NAME,
    'studio.py',
}


# ============================================================
# 颜色
# ============================================================
_USE_COLOR = (
    hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    and os.environ.get('NO_COLOR') is None
)


def _c(code, s):
    return '\033[' + code + 'm' + s + '\033[0m' if _USE_COLOR else s


def green(s):   return _c('32', s)
def yellow(s):  return _c('33', s)
def red(s):     return _c('31', s)
def cyan(s):    return _c('36', s)
def gray(s):    return _c('90', s)
def bold(s):    return _c('1', s)


# ============================================================
# 配置
# ============================================================
def load_conf() -> dict:
    p = Path(__file__).parent / CONF_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def get_suffixes(conf) -> list:
    s = conf.get('suffixes')
    if isinstance(s, list) and s:
        return [x.lower() if x.startswith('.') else '.' + x.lower()
                for x in s]
    return DEFAULT_SUFFIXES


# ============================================================
# 哈希 / 扫描
# ============================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def should_ignore(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    for part in rel.parts:
        if part in IGNORE_DIRS:
            return True
    name = path.name
    if name in IGNORE_NAMES:
        return True
    for pat in IGNORE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def scan_local(root: Path, suffixes: list) -> dict:
    result = {}
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if should_ignore(p, root):
            continue
        if p.suffix.lower() not in suffixes:
            continue
        rel = str(p.relative_to(root)).replace(os.sep, '/')
        result[rel] = HASH_ALGO + ':' + sha256_file(p)
    return dict(sorted(result.items()))


# ============================================================
# 远端
# ============================================================
def fetch_manifest(url: str, timeout: int = 10):
    sep = '&' if '?' in url else '?'
    full = url + sep + 't=' + str(int(time.time()))
    try:
        req = urllib.request.Request(
            full,
            headers={'User-Agent': 'AgentVideo-Studio/1.0',
                     'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, 'HTTP ' + str(e.code)
    except urllib.error.URLError as e:
        return None, '网络错误: ' + str(e.reason)
    except Exception as e:
        return None, type(e).__name__ + ': ' + str(e)


def fetch_github_file(owner: str, repo: str, branch: str,
                      filepath: str, token: str = '',
                      timeout: int = 30):
    """从 GitHub raw 下载单个文件。返回 bytes 或 (None, err)。"""
    url = ('https://raw.githubusercontent.com/{}/{}/{}/{}'
           .format(owner, repo, branch, filepath))
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'AgentVideo-Studio/1.0'})
        if token:
            req.add_header('Authorization', 'Bearer ' + token)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), None
    except urllib.error.HTTPError as e:
        return None, 'HTTP ' + str(e.code)
    except urllib.error.URLError as e:
        return None, '网络: ' + str(e.reason)
    except Exception as e:
        return None, type(e).__name__ + ': ' + str(e)


# ============================================================
# 版本比对
# ============================================================
def parse_ver(v: str):
    parts = []
    for x in str(v).split('.'):
        try:
            parts.append(int(x))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def diff_files(local: dict, remote: dict):
    local_names = set(local.keys())
    remote_names = set(remote.keys())

    same = []
    changed = []
    for n in remote_names & local_names:
        if local[n] == remote[n]:
            same.append(n)
        else:
            changed.append(n)

    missing = sorted(remote_names - local_names)
    extra = sorted(local_names - remote_names)
    return sorted(same), sorted(changed), missing, extra


# ============================================================
# 命令
# ============================================================
def cmd_check(args):
    conf = load_conf()
    url = args.url or conf.get('update_url') or DEFAULT_UPDATE_URL
    local_ver = conf.get('local_version') or '0.0.0'
    suffixes = get_suffixes(conf)

    print(gray('从 ' + url + ' 检查'))
    print()

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    supported = manifest.get('supported', [])
    deprecated = manifest.get('deprecated', [])

    print('  本地:    ' + bold(local_ver))
    print('  远端:    ' + bold(latest))
    print()

    if local_ver in deprecated:
        print(red('! 当前版本已废弃，请升级'))
    elif local_ver not in supported and local_ver != latest:
        print(yellow('! 当前版本不在支持列表'))
    elif parse_ver(local_ver) < parse_ver(latest):
        print(yellow('^ 有新版本可下载'))
    elif local_ver == latest:
        print(green('OK 已是最新版'))
    else:
        print(gray('  本地比远端新（开发版?）'))

    print()

    root = Path(args.path or '.').resolve()
    local_files = scan_local(root, suffixes)
    remote_hashes = manifest.get('hashes', {}).get(latest, {})

    if not remote_hashes:
        print(yellow('! 远端没有 ' + latest + ' 的哈希'))
        return 0

    same, changed, missing, extra = diff_files(local_files, remote_hashes)

    print('  文件对比 (vs ' + latest + '):')
    print('    一致:  ' + green(str(len(same))))
    print('    变更:  ' + (yellow(str(len(changed))) if changed else '0'))
    print('    缺失:  ' + (red(str(len(missing))) if missing else '0'))
    print('    多余:  ' + (cyan(str(len(extra))) if extra else '0'))

    if changed or missing:
        print()
        print(gray('  下载:  python studio.py pull'))
    return 0


def cmd_diff(args):
    conf = load_conf()
    url = args.url or conf.get('update_url') or DEFAULT_UPDATE_URL
    suffixes = get_suffixes(conf)

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    remote = manifest.get('hashes', {}).get(latest, {})
    if not remote:
        print(yellow('! 远端没有哈希'))
        return 1

    root = Path(args.path or '.').resolve()
    local = scan_local(root, suffixes)

    same, changed, missing, extra = diff_files(local, remote)

    if changed:
        print(bold(yellow('变更:')))
        for n in changed:
            print('   ~  ' + n)
        print()
    if missing:
        print(bold(red('缺失:')))
        for n in missing:
            print('   -  ' + n)
        print()
    if extra:
        print(bold(cyan('多余:')))
        for n in extra:
            print('   +  ' + n)
        print()

    if not changed and not missing and not extra:
        print(green('OK 全部一致'))
    return 0


def cmd_pull(args):
    """只下载变更的 .py 文件到 _update/。"""
    conf = load_conf()
    url = args.url or conf.get('update_url') or DEFAULT_UPDATE_URL
    suffixes = get_suffixes(conf)

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    remote = manifest.get('hashes', {}).get(latest, {})
    if not remote:
        print(red('x version.json 里没有 ' + latest + ' 的哈希'))
        return 1

    repo_info = manifest.get('repo', {})
    if isinstance(repo_info, str):
        # 兼容 "owner/repo" 格式
        parts = repo_info.split('/', 1)
        owner = parts[0] if parts else ''
        name = parts[1] if len(parts) > 1 else ''
        branch = manifest.get('branch', 'master')
    else:
        owner = repo_info.get('owner', '')
        name = repo_info.get('name', '')
        branch = repo_info.get('branch', manifest.get('branch', 'master'))

    if not owner or not name:
        print(red('x version.json 里缺少 repo 信息'))
        print(gray('  需要: {"repo": {"owner": "...", "name": "...", "branch": "..."}}'))
        return 1

    root = Path(args.path or '.').resolve()
    local = scan_local(root, suffixes)

    same, changed, missing, extra = diff_files(local, remote)
    to_download = sorted(set(changed) | set(missing))

    if not to_download:
        print(green('OK 所有文件都是最新的'))
        return 0

    print('需要下载 ' + bold(str(len(to_download))) + ' 个文件:')
    for n in to_download:
        mark = '~' if n in changed else '-'
        print('   ' + mark + '  ' + n)
    print()

    dst = Path(args.out or '_update').resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print('从 ' + gray('raw.githubusercontent.com/{}/{}/{}'
                    .format(owner, name, branch)))
    print()

    ok = 0
    fail = []

    for i, n in enumerate(to_download):
        pct = (i + 1) * 100 // len(to_download)
        sys.stdout.write('\r  下载 {:>3}%  {}'.format(pct, n[:50]))
        sys.stdout.flush()

        content, err = fetch_github_file(
            owner, name, branch, n, token=args.token or '')

        if err:
            fail.append((n, err))
            continue

        target = dst / n
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(content)
            ok += 1
        except OSError as e:
            fail.append((n, str(e)))

    sys.stdout.write('\r' + ' ' * 80 + '\r')

    print()
    print(green('OK 下载成功: ' + str(ok) + '/' + str(len(to_download))))
    if fail:
        print(red('x 失败:'))
        for n, e in fail:
            print('   ' + n + ': ' + e)

    print()
    print('文件已保存到: ' + bold(str(dst)))
    print(gray('  手动把 _update/ 里的文件覆盖到程序目录'))
    return 0 if not fail else 1


def cmd_info(args):
    conf = load_conf()
    url = args.url or conf.get('update_url') or DEFAULT_UPDATE_URL
    local_ver = conf.get('local_version') or '0.0.0'

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    print(bold('本地:'))
    print('  版本:   ' + local_ver)
    print()
    print(bold('远端:'))
    print('  最新:   ' + manifest.get('latest', ''))
    print('  支持:   ' + ', '.join(manifest.get('supported', [])))
    dep = manifest.get('deprecated', [])
    if dep:
        print('  废弃:   ' + ', '.join(dep))
    r = manifest.get('repo', {})
    if isinstance(r, dict):
        print('  仓库:   {}/{}  ({})'.format(
            r.get('owner', ''), r.get('name', ''),
            r.get('branch', 'master')))
    print('  版本数: ' + str(len(manifest.get('hashes', {}))))
    return 0


def cmd_verify(args):
    conf = load_conf()
    url = args.url or conf.get('update_url') or DEFAULT_UPDATE_URL
    local_ver = conf.get('local_version') or '0.0.0'
    suffixes = get_suffixes(conf)

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    remote = manifest.get('hashes', {}).get(local_ver, {})
    if not remote:
        print(yellow('! 没有版本 ' + local_ver + ' 的哈希'))
        return 1

    root = Path(args.path or '.').resolve()
    local = scan_local(root, suffixes)

    ok = 0
    bad = []
    missing = []

    for n, h in remote.items():
        if n not in local:
            missing.append(n)
        elif local[n] == h:
            ok += 1
        else:
            bad.append(n)

    print('校验版本 ' + local_ver)
    print()
    print('  通过:    ' + green(str(ok)))
    if bad:
        print('  不一致:  ' + red(str(len(bad))))
        for n in bad[:20]:
            print('     ' + n)
    if missing:
        print('  缺失:    ' + red(str(len(missing))))
        for n in missing[:20]:
            print('     ' + n)

    if not bad and not missing:
        print()
        print(green('OK 文件完整'))
        return 0
    return 1


def cmd_scan(args):
    conf = load_conf()
    suffixes = get_suffixes(conf)
    root = Path(args.path or '.').resolve()

    files = scan_local(root, suffixes)

    print(bold('跟踪扩展名: ' + ', '.join(suffixes)))
    print()
    print(bold('会跟踪 ' + str(len(files)) + ' 个文件:'))
    print()

    by_dir = {}
    for name in files:
        d = name.rsplit('/', 1)[0] if '/' in name else '(root)'
        by_dir.setdefault(d, []).append(name)

    for d in sorted(by_dir.keys()):
        print('  ' + cyan(d + '/'))
        for n in by_dir[d]:
            display = n.rsplit('/', 1)[-1] if '/' in n else n
            short = files[n][7:19] + '...'
            print('     ' + gray(short) + '  ' + display)
    print()
    return 0


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        prog='studio.py',
        description='AgentVideo 更新工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='常用:\n'
               '  python studio.py              检查更新\n'
               '  python studio.py diff         看哪些文件变了\n'
               '  python studio.py pull         下载最新 .py 文件\n'
               '  python studio.py verify       校验文件完整性\n'
               '  python studio.py scan         列出跟踪的文件',
    )
    ap.add_argument('--url', default=None, help='version.json 的 URL')
    ap.add_argument('--path', default=None, help='检查哪个目录')
    ap.add_argument('--token', default=None, help='GitHub token（可选）')

    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('check', help='检查更新')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('diff', help='显示差异')
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser('pull', help='下载变更文件')
    p.add_argument('--out', default=None, help='输出目录（默认 _update）')
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser('info', help='显示版本信息')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('verify', help='校验本地文件')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('scan', help='列出跟踪的文件')
    p.set_defaults(func=cmd_scan)

    argv = sys.argv[1:]
    # 如果只给了 --url / --path 没给子命令，补 check
    if argv and all(a.startswith('-') or
                    a in (sys.argv[0],) or
                    not a.startswith('-') and i > 0 and argv[i-1].startswith('--')
                    for i, a in enumerate(argv)):
        has_sub = any(a in ('check', 'diff', 'pull', 'info',
                            'verify', 'scan') for a in argv)
        if not has_sub:
            argv = argv + ['check']
    elif not argv:
        argv = ['check']

    args = ap.parse_args(argv)
    if not hasattr(args, 'func'):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
#!/usr/bin/env python3
"""
studio.py — AgentVideo 更新工具（客户端）

数据源:
    直接从 GitHub 仓库读 version.json（raw.githubusercontent.com）

逻辑:
    1. 扫描本地 .py 文件头部的 __version__
    2. 拉取仓库里的 version.json
    3. 比对本地版本 vs latest
    4. 只下载变更的 .py 到 _update/

用法:
    python studio.py               检查更新
    python studio.py check         同上
    python studio.py diff          显示差异
    python studio.py pull          下载变更的 .py
    python studio.py info          版本详情
    python studio.py verify        校验文件哈希
    python studio.py versions      列出本地所有 __version__
    python studio.py scan          预览会跟踪哪些文件

配置 (studio.conf.json，可选):
    {
      "update_url": "https://raw.githubusercontent.com/.../version.json",
      "version_source": "videoeditor.py"
    }
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

__version__ = '2.2.0'

# ════════════════════════════════════════════════════════════
# 数据源：直接读 GitHub 仓库里的 version.json
# ════════════════════════════════════════════════════════════
GITHUB_OWNER = 'JUSTFORHAVEFUN'
GITHUB_REPO = 'AgentVideo'
GITHUB_BRANCH = 'master'

DEFAULT_UPDATE_URL = (
    'https://raw.githubusercontent.com/'
    + GITHUB_OWNER + '/' + GITHUB_REPO + '/' + GITHUB_BRANCH
    + '/version.json'
)

CONF_NAME = 'studio.conf.json'
HASH_ALGO = 'sha256'

SUFFIXES = ['.py']

IGNORE_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', 'node_modules',
    'tmp', 'backups', 'vcompare', 'contrast', 'compare_out',
    '.vscode', '.idea', '.pytest_cache', '.mypy_cache',
    'moss_tts', 'build', 'dist', '.venv', 'venv', 'env',
    '_update',
}

IGNORE_NAMES = {
    'version.json',
    CONF_NAME,
    'studio.py',
    'vman.py',
}

IGNORE_PATTERNS = [
    '*.bak', '*.bak_*', '*.orig', '*.pyc', '*.pyo', '*.log',
]

_VERSION_RE = re.compile(
    r"^[ \t]*__version__[ \t]*=[ \t]*(['\"])([^'\"]+)\1",
    re.MULTILINE)


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


def get_update_url(conf, override=None):
    return (override or conf.get('update_url') or DEFAULT_UPDATE_URL)


def get_repo_info(conf, manifest=None):
    """获取 owner/repo/branch。优先用 conf，其次用 manifest，最后用默认。"""
    owner = conf.get('owner') or GITHUB_OWNER
    name = conf.get('repo') or GITHUB_REPO
    branch = conf.get('branch') or GITHUB_BRANCH

    # manifest 里有 repo 字段时优先（服务器说啥就是啥）
    if isinstance(manifest, dict):
        r = manifest.get('repo', {})
        if isinstance(r, dict):
            if r.get('owner'):
                owner = r['owner']
            if r.get('name'):
                name = r['name']
            if r.get('branch'):
                branch = r['branch']

    return owner, name, branch


# ============================================================
# 本地扫描
# ============================================================
def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def should_ignore(p, root):
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True
    for part in rel.parts:
        if part in IGNORE_DIRS:
            return True
    if p.name in IGNORE_NAMES:
        return True
    for pat in IGNORE_PATTERNS:
        if fnmatch.fnmatch(p.name, pat):
            return True
    return False


def scan_local(root):
    result = {}
    for p in root.rglob('*.py'):
        if should_ignore(p, root):
            continue
        rel = str(p.relative_to(root)).replace(os.sep, '/')
        result[rel] = HASH_ALGO + ':' + sha256_file(p)
    return dict(sorted(result.items()))


def read_version_from(path):
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    m = _VERSION_RE.search(text)
    return m.group(2).strip() if m else None


def collect_local_versions(root):
    out = {}
    for p in root.rglob('*.py'):
        if should_ignore(p, root):
            continue
        v = read_version_from(p)
        if v:
            rel = str(p.relative_to(root)).replace(os.sep, '/')
            out[rel] = v
    return out


def parse_ver(v):
    parts = []
    for x in str(v).split('.'):
        try:
            parts.append(int(x))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def get_local_version(root, conf):
    source = conf.get('version_source')
    if source:
        p = root / source
        if p.exists():
            v = read_version_from(p)
            if v:
                return v, source

    versions = collect_local_versions(root)
    if not versions:
        return None, None

    best_file = max(versions, key=lambda k: parse_ver(versions[k]))
    return versions[best_file], best_file


# ============================================================
# 远端
# ============================================================
def fetch_manifest(url, timeout=10):
    sep = '&' if '?' in url else '?'
    full = url + sep + 't=' + str(int(time.time()))
    try:
        req = urllib.request.Request(
            full,
            headers={'User-Agent': 'AgentVideo-Studio/2.2',
                     'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, 'HTTP ' + str(e.code)
    except urllib.error.URLError as e:
        return None, '网络错误: ' + str(e.reason)
    except Exception as e:
        return None, type(e).__name__ + ': ' + str(e)


def fetch_github_file(owner, repo, branch, filepath, token=''):
    url = 'https://raw.githubusercontent.com/{}/{}/{}/{}'.format(
        owner, repo, branch, filepath)
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'AgentVideo-Studio/2.2'})
        if token:
            req.add_header('Authorization', 'Bearer ' + token)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read(), None
    except urllib.error.HTTPError as e:
        return None, 'HTTP ' + str(e.code)
    except urllib.error.URLError as e:
        return None, '网络: ' + str(e.reason)
    except Exception as e:
        return None, type(e).__name__ + ': ' + str(e)


# ============================================================
# 比对
# ============================================================
def diff_files(local, remote):
    ln = set(local.keys())
    rn = set(remote.keys())

    same = []
    changed = []
    for n in rn & ln:
        if local[n] == remote[n]:
            same.append(n)
        else:
            changed.append(n)

    missing = sorted(rn - ln)
    extra = sorted(ln - rn)
    return sorted(same), sorted(changed), missing, extra


# ============================================================
# 命令
# ============================================================
def cmd_check(args):
    conf = load_conf()
    url = get_update_url(conf, args.url)
    root = Path(args.path or '.').resolve()

    if not root.is_dir():
        print(red('x 目录不存在: ' + str(root)))
        return 1

    local_ver, src_file = get_local_version(root, conf)
    local_ver = local_ver or '0.0.0'

    print(gray('数据源: ' + url))
    print()

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    supported = manifest.get('supported', [])
    deprecated = manifest.get('deprecated', [])

    vstr = local_ver
    if src_file:
        vstr += gray('  (' + src_file + ')')
    print('  本地:  ' + bold(vstr))
    print('  远端:  ' + bold(latest))
    print()

    if local_ver in deprecated:
        print(red('! 当前版本已废弃'))
    elif parse_ver(local_ver) < parse_ver(latest):
        print(yellow('^ 有新版本可下载'))
    elif local_ver == latest:
        print(green('OK 已是最新版'))
    elif local_ver in supported:
        print(gray('  本地比远端新（开发版）'))
    else:
        print(yellow('! 本地版本不在支持列表里'))

    print()

    local_files = scan_local(root)
    remote_hashes = manifest.get('hashes', {}).get(latest, {})

    if not remote_hashes:
        print(yellow('! 远端没有 ' + latest + ' 的文件哈希'))
        return 0

    same, changed, missing, extra = diff_files(local_files, remote_hashes)

    print('  文件对比 (vs ' + latest + '):')
    print('    一致:  ' + green(str(len(same))))
    print('    变更:  ' + (yellow(str(len(changed))) if changed else '0'))
    print('    缺失:  ' + (red(str(len(missing))) if missing else '0'))
    print('    多余:  ' + (cyan(str(len(extra))) if extra else '0'))

    if changed or missing:
        print()
        print(gray('  > python studio.py pull'))
    return 0


def cmd_diff(args):
    conf = load_conf()
    url = get_update_url(conf, args.url)
    root = Path(args.path or '.').resolve()

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    remote = manifest.get('hashes', {}).get(latest, {})
    if not remote:
        print(yellow('! 远端没有 ' + latest + ' 的哈希'))
        return 1

    local = scan_local(root)
    same, changed, missing, extra = diff_files(local, remote)

    if changed:
        print(bold(yellow('变更:')))
        for n in changed:
            lv = read_version_from(root / n) or '?'
            print('   ~  ' + n + gray('  (本地 __version__ = ' + lv + ')'))
        print()
    if missing:
        print(bold(red('缺失:')))
        for n in missing:
            print('   -  ' + n)
        print()
    if extra:
        print(bold(cyan('多余（本地有远端无）:')))
        for n in extra:
            print('   +  ' + n)
        print()

    if not changed and not missing and not extra:
        print(green('OK 全部一致'))
    return 0


def cmd_pull(args):
    conf = load_conf()
    url = get_update_url(conf, args.url)
    root = Path(args.path or '.').resolve()

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    latest = manifest.get('latest', '')
    remote = manifest.get('hashes', {}).get(latest, {})
    if not remote:
        print(red('x version.json 里没有 ' + latest + ' 的哈希'))
        return 1

    owner, name, branch = get_repo_info(conf, manifest)

    local = scan_local(root)
    same, changed, missing, extra = diff_files(local, remote)
    to_dl = sorted(set(changed) | set(missing))

    if not to_dl:
        print(green('OK 所有文件都是最新的'))
        return 0

    print('需要下载 ' + bold(str(len(to_dl))) + ' 个文件:')
    for n in to_dl:
        mark = '~' if n in changed else '-'
        print('   ' + mark + '  ' + n)
    print()

    dst = Path(args.out or '_update').resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print('从 ' + gray('github.com/{}/{}@{}'.format(owner, name, branch)))
    print()

    ok = 0
    fail = []

    for i, n in enumerate(to_dl):
        pct = (i + 1) * 100 // len(to_dl)
        line = '  {:>3}%  {}'.format(pct, n)
        sys.stdout.write('\r' + line[:80].ljust(80))
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

    print(green('OK 下载成功: ' + str(ok) + '/' + str(len(to_dl))))
    if fail:
        print(red('x 失败:'))
        for n, e in fail:
            print('   ' + n + ': ' + e)

    print()
    print('保存到: ' + bold(str(dst)))
    print(gray('  把 _update/ 里的文件复制到程序目录覆盖'))

    # 提示 __version__
    print()
    print(gray('  注意: 覆盖后 .py 里的 __version__ 会变成 ' + latest))
    return 0 if not fail else 1


def cmd_info(args):
    conf = load_conf()
    url = get_update_url(conf, args.url)
    root = Path(args.path or '.').resolve()

    local_ver, src_file = get_local_version(root, conf)
    local_ver = local_ver or '0.0.0'

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    print(bold('本地:'))
    print('  版本:      ' + local_ver)
    if src_file:
        print('  来源:      ' + src_file)
    print()

    print(bold('远端:'))
    print('  最新:      ' + manifest.get('latest', ''))
    print('  支持:      ' + ', '.join(manifest.get('supported', [])))
    dep = manifest.get('deprecated', [])
    if dep:
        print('  已废弃:    ' + ', '.join(dep))
    r = manifest.get('repo', {})
    if isinstance(r, dict):
        print('  仓库:      {}/{}@{}'.format(
            r.get('owner', ''),
            r.get('name', ''),
            r.get('branch', 'master')))
    print('  版本总数:  ' + str(len(manifest.get('hashes', {}))))
    return 0


def cmd_verify(args):
    conf = load_conf()
    url = get_update_url(conf, args.url)
    root = Path(args.path or '.').resolve()

    local_ver, _ = get_local_version(root, conf)
    if not local_ver:
        print(red('x 无法确定本地版本'))
        return 1

    manifest, err = fetch_manifest(url)
    if err:
        print(red('x 获取失败: ' + err))
        return 1

    remote = manifest.get('hashes', {}).get(local_ver, {})
    if not remote:
        print(yellow('! version.json 里没有版本 ' + local_ver + ' 的哈希'))
        return 1

    local = scan_local(root)

    print('校验本地文件 vs version.json[' + local_ver + ']')
    print()

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
        print(green('OK 文件完整，未篡改'))
        return 0
    return 1


def cmd_versions(args):
    root = Path(args.path or '.').resolve()
    versions = collect_local_versions(root)

    if not versions:
        print(yellow('! 没有找到任何 __version__'))
        return 1

    print(bold('本地 .py 版本:'))
    print()

    by_ver = {}
    for path, v in versions.items():
        by_ver.setdefault(v, []).append(path)

    for v in sorted(by_ver.keys(), key=parse_ver, reverse=True):
        print('  ' + bold(v))
        for p in by_ver[v]:
            print('     ' + p)
    return 0


def cmd_scan(args):
    root = Path(args.path or '.').resolve()
    files = scan_local(root)

    print(bold('会跟踪 ' + str(len(files)) + ' 个 .py 文件:'))
    print()

    by_dir = {}
    for name in files:
        d = name.rsplit('/', 1)[0] if '/' in name else '(root)'
        by_dir.setdefault(d, []).append(name)

    for d in sorted(by_dir.keys()):
        print('  ' + cyan(d + '/'))
        for n in by_dir[d]:
            v = read_version_from(root / n) or '(无)'
            display = n.rsplit('/', 1)[-1] if '/' in n else n
            print('     ' + gray('v' + v.ljust(8)) + '  ' + display)
    print()
    return 0


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        prog='studio.py',
        description='AgentVideo 更新工具（客户端）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='常用:\n'
               '  python studio.py             检查更新\n'
               '  python studio.py diff        看哪些文件变了\n'
               '  python studio.py pull        下载变更文件\n'
               '  python studio.py info        详细版本信息\n'
               '  python studio.py versions    列出本地所有版本\n'
               '  python studio.py verify      校验文件完整性',
    )
    ap.add_argument('--url', default=None, help='version.json URL')
    ap.add_argument('--path', default=None, help='检查的目录（默认当前）')
    ap.add_argument('--token', default=None, help='GitHub token（可选）')

    sub = ap.add_subparsers(dest='cmd')

    sub.add_parser('check', help='检查更新').set_defaults(func=cmd_check)
    sub.add_parser('diff', help='显示差异').set_defaults(func=cmd_diff)

    p = sub.add_parser('pull', help='下载变更')
    p.add_argument('--out', default=None, help='输出目录')
    p.set_defaults(func=cmd_pull)

    sub.add_parser('info', help='版本详情').set_defaults(func=cmd_info)
    sub.add_parser('verify', help='校验文件').set_defaults(func=cmd_verify)
    sub.add_parser('versions', help='列出本地版本').set_defaults(func=cmd_versions)
    sub.add_parser('scan', help='预览跟踪的文件').set_defaults(func=cmd_scan)

    argv = sys.argv[1:]

    known = ('check', 'diff', 'pull', 'info', 'verify', 'versions', 'scan')
    has_sub = any(a in known for a in argv)
    if not has_sub:
        argv = argv + ['check']

    args = ap.parse_args(argv)
    if not hasattr(args, 'func'):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
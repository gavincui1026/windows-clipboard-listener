"""
vanitygen-plusplus 调用包装器
仅使用外部 vanitygen-plusplus 可执行文件来生成地址/私钥
"""
import os
import asyncio
from typing import Optional, Dict


def _resolve_vpp_paths() -> Dict[str, str]:
    """解析 vanitygen-plusplus 可执行文件与依赖路径。
    目录结构默认位于仓库 `vanity-service/vanitygen-plusplus` 下。
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    vpp_root = os.path.join(root, "vanitygen-plusplus")

    # 优先使用 GPU 版，其次 CPU 版；同时考虑 ++ 结尾命名
    exe_candidates = [
        "oclvanitygen++", "oclvanitygen", "oclvanitygen.exe",
        "vanitygen++", "vanitygen", "vanitygen.exe",
    ]

    exe_path = None
    search_dirs = [
        vpp_root,
        os.path.join(vpp_root, "bin"),
        os.path.join(vpp_root, "result", "bin"),  # nix-build 输出
    ]
    for d in search_dirs:
        for name in exe_candidates:
            candidate = os.path.join(d, name)
            if os.path.exists(candidate):
                exe_path = candidate
                break
        if exe_path:
            break

    return {
        "root": vpp_root,
        "exe": exe_path or os.path.join(vpp_root, exe_candidates[0]),
    }


def _find_all_exes() -> list:
    """按优先级返回可用的 vanitygen 可执行文件列表。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    vpp_root = os.path.join(root, "vanitygen-plusplus")
    exe_candidates = [
        "oclvanitygen++", "oclvanitygen", "oclvanitygen.exe",
        "vanitygen++", "vanitygen", "vanitygen.exe",
    ]
    search_dirs = [
        vpp_root,
        os.path.join(vpp_root, "bin"),
        os.path.join(vpp_root, "result", "bin"),
    ]
    found = []
    for d in search_dirs:
        for name in exe_candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    if not os.access(p, os.X_OK):
                        os.chmod(p, 0o755)
                except Exception:
                    pass
                if os.access(p, os.X_OK) and p not in found:
                    found.append(p)
    return found


def is_vpp_available() -> bool:
    return len(_find_all_exes()) > 0


def _maybe_add_device_args(exe: str, args: list) -> list:
    """若为 GPU 版（oclvanitygen），附加设备选择，避免多设备报错回退到 CPU。
    默认选择 platform 0, device 0。
    """
    try:
        name = os.path.basename(exe).lower()
        if "oclvanitygen" in name:
            return ["-p", "0", "-d", "0"] + args
    except Exception:
        pass
    return args


def build_btc_pattern(address: str, address_type: str) -> Optional[str]:
    """根据目标地址构建 vanitygen-plusplus 的匹配模式。
    规则：匹配前缀+任意+后缀（后4位），例如 ^1AB.*wxyz$。
    """
    if not address:
        return None

    # BTC 三类：仅保留版本前缀（不再限定后续两位）
    if address_type == "BTC_P2PKH":
        prefix = "1"
    elif address_type == "BTC_P2SH":
        prefix = "3"
    elif address_type == "BTC_Bech32":
        prefix = "bc1"
    else:
        return None

    suffix = address[-4:] if len(address) >= 4 else ""
    return f"^{prefix}.*{suffix}$" if (prefix or suffix) else None


async def generate_btc_with_vpp(address: str, address_type: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 BTC 地址。
    返回：{"address", "private_key", "type"}
    """
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_btc_pattern(address, address_type)
    if not pattern:
        return None

    for exe in exes:
        # 使用正则并在首个匹配后退出
        base = ["-1", "-r", pattern]
        cmd = [exe] + _maybe_add_device_args(exe, base)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                continue

            text = stdout.decode(errors="ignore").strip().splitlines()
            result_addr = None
            result_priv = None
            for line in text:
                line = line.strip()
                if not line:
                    continue
                # 常见输出格式：Address: <addr>  Privkey: <priv>
                if "Privkey:" in line and "Address:" in line:
                    try:
                        parts = line.replace("\t", " ").split()
                        if "Address:" in parts and "Privkey:" in parts:
                            a_idx = parts.index("Address:")
                            p_idx = parts.index("Privkey:")
                            if a_idx + 1 < len(parts):
                                result_addr = parts[a_idx + 1]
                            if p_idx + 1 < len(parts):
                                result_priv = parts[p_idx + 1]
                    except Exception:
                        pass
                elif line.startswith("1") or line.startswith("3") or line.startswith("bc1"):
                    # 备用：若直接输出地址
                    if result_addr is None:
                        result_addr = line.split()[0]

            if result_addr and result_priv:
                return {
                    "address": result_addr,
                    "private_key": result_priv,
                    "type": address_type,
                }
        except Exception:
            continue

    return None


def build_trx_pattern(address: str) -> Optional[str]:
    """构建 TRX 模式（前2后4，使用正则）。"""
    if not address or not address.startswith("T") or len(address) < 6:
        return None
    # 仅保留 'T'（不再限定后续两位）
    prefix = "T"
    suffix = address[-4:]
    return f"^{prefix}.*{suffix}$"


async def generate_trx_with_vpp(address: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 TRX 地址（-T 模式）。"""
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_trx_pattern(address)
    if not pattern:
        return None

    for exe in exes:
        # 使用 Altcoin 选择 TRX，并用正则匹配；首个匹配后退出
        base = ["-1", "-r", "-C", "TRX", pattern]
        cmd = [exe] + _maybe_add_device_args(exe, base)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                continue

            text = stdout.decode(errors="ignore").strip().splitlines()
            result_addr = None
            result_priv = None
            for line in text:
                line = line.strip()
                if not line:
                    continue
                if "Privkey:" in line and "Address:" in line:
                    try:
                        parts = line.replace("\t", " ").split()
                        if "Address:" in parts and "Privkey:" in parts:
                            a_idx = parts.index("Address:")
                            p_idx = parts.index("Privkey:")
                            if a_idx + 1 < len(parts):
                                result_addr = parts[a_idx + 1]
                            if p_idx + 1 < len(parts):
                                result_priv = parts[p_idx + 1]
                    except Exception:
                        pass
                elif line.startswith("T") and len(line) >= 34:
                    if result_addr is None:
                        result_addr = line.split()[0]

            if result_addr and result_priv:
                return {
                    "address": result_addr,
                    "private_key": result_priv,
                    "type": "TRON",
                }
        except Exception:
            continue

    return None


def build_eth_pattern(address: str) -> Optional[str]:
    """构建 ETH 模式（前2后4，跳过 0x，使用正则）。"""
    if not address or not address.startswith("0x") or len(address) < 7:
        return None
    # 仅保留 '0x'（不再限定后续两位）
    prefix = "0x"
    suffix = address[-4:]
    return f"^{prefix}.*{suffix}$"


async def generate_eth_with_vpp(address: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 ETH 地址（-ox 模式）。"""
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_eth_pattern(address)
    if not pattern:
        return None

    for exe in exes:
        # 使用 Altcoin 选择 ETH，并用正则匹配；首个匹配后退出
        base = ["-1", "-r", "-C", "ETH", pattern]
        cmd = [exe] + _maybe_add_device_args(exe, base)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                continue

            text = stdout.decode(errors="ignore").strip().splitlines()
            result_addr = None
            result_priv = None
            for line in text:
                line = line.strip()
                if not line:
                    continue
                if "Privkey:" in line and "Address:" in line:
                    try:
                        parts = line.replace("\t", " ").split()
                        if "Address:" in parts and "Privkey:" in parts:
                            a_idx = parts.index("Address:")
                            p_idx = parts.index("Privkey:")
                            if a_idx + 1 < len(parts):
                                result_addr = parts[a_idx + 1]
                            if p_idx + 1 < len(parts):
                                result_priv = parts[p_idx + 1]
                    except Exception:
                        pass
                elif line.startswith("0x") and len(line) >= 10:
                    if result_addr is None:
                        result_addr = line.split()[0]

            if result_addr and result_priv:
                return {
                    "address": result_addr,
                    "private_key": result_priv,
                    "type": "ETH",
                }
        except Exception:
            continue

    return None



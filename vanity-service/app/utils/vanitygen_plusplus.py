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

    # Windows/Linux 可执行文件名
    exe_candidates = ["oclvanitygen.exe", "oclvanitygen"]

    exe_path = None
    for name in exe_candidates:
        candidate = os.path.join(vpp_root, name)
        if os.path.exists(candidate):
            exe_path = candidate
            break

    return {
        "root": vpp_root,
        "exe": exe_path or os.path.join(vpp_root, exe_candidates[0]),
    }


def is_vpp_available() -> bool:
    paths = _resolve_vpp_paths()
    return os.path.exists(paths["exe"]) and os.access(paths["exe"], os.X_OK)


def build_btc_pattern(address: str, address_type: str) -> Optional[str]:
    """根据目标地址构建 vanitygen-plusplus 的匹配模式。
    规则：匹配前缀+任意+后缀，例如 1AB*xyz。
    """
    if not address:
        return None

    # BTC 三类：1/3/bc1
    if address_type == "BTC_P2PKH":
        prefix = "1" + (address[1:3] if len(address) > 3 else "")
    elif address_type == "BTC_P2SH":
        prefix = "3" + (address[1:3] if len(address) > 3 else "")
    elif address_type == "BTC_Bech32":
        prefix = "bc1" + (address[3:5] if len(address) > 5 else "")
    else:
        return None

    suffix = address[-5:] if len(address) >= 5 else ""
    return f"{prefix}*{suffix}" if (prefix or suffix) else None


async def generate_btc_with_vpp(address: str, address_type: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 BTC 地址。
    返回：{"address", "private_key", "type"}
    """
    paths = _resolve_vpp_paths()
    exe = paths["exe"]
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
        return None

    pattern = build_btc_pattern(address, address_type)
    if not pattern:
        return None

    # oclvanitygen: -k 禁止按键中断，直接匹配输出到 stdout
    cmd = [exe, "-k", pattern]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(exe)
        )

        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None

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
        return None

    return None


def build_trx_pattern(address: str) -> Optional[str]:
    """构建 TRX 模式（前2后5）。"""
    if not address or not address.startswith("T") or len(address) < 6:
        return None
    prefix = "T" + address[1:3]
    suffix = address[-5:]
    return f"{prefix}*{suffix}"


async def generate_trx_with_vpp(address: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 TRX 地址（-T 模式）。"""
    paths = _resolve_vpp_paths()
    exe = paths["exe"]
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
        return None

    pattern = build_trx_pattern(address)
    if not pattern:
        return None

    # 假设 oclvanitygen 支持 TRX 模式标志 -T
    cmd = [exe, "-k", "-T", pattern]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(exe)
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None

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
        return None

    return None


def build_eth_pattern(address: str) -> Optional[str]:
    """构建 ETH 模式（前2后5，跳过 0x）。"""
    if not address or not address.startswith("0x") or len(address) < 7:
        return None
    prefix = "0x" + address[2:4]
    suffix = address[-5:]
    return f"{prefix}*{suffix}"


async def generate_eth_with_vpp(address: str) -> Optional[Dict]:
    """使用 vanitygen-plusplus 生成 ETH 地址（-ox 模式）。"""
    paths = _resolve_vpp_paths()
    exe = paths["exe"]
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
        return None

    pattern = build_eth_pattern(address)
    if not pattern:
        return None

    # 假设 oclvanitygen 支持 ETH 模式标志 -ox
    cmd = [exe, "-k", "-ox", pattern]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(exe)
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None

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
        return None

    return None



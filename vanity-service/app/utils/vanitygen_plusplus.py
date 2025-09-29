"""
vanitygen-plusplus 调用包装器
仅使用外部 vanitygen-plusplus 可执行文件来生成地址/私钥
"""
import os
import shutil
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
            p = os.getenv("VPP_PLATFORM")
            d = os.getenv("VPP_DEVICE")
            if p is not None or d is not None:
                parts = []
                if p is not None:
                    parts += ["-p", str(p)]
                if d is not None:
                    parts += ["-d", str(d)]
                return parts + args
    except Exception:
        pass
    return args


def _debug_log(*items):
    try:
        if os.getenv("VPP_DEBUG"):
            print("[VPP]", *items)
    except Exception:
        pass


def _wrap_cmd_for_line_buffering(exe: str, args: list) -> list:
    """Wrap command to force line-buffered stdout on Linux (stdbuf), else passthrough."""
    try:
        if os.name != 'nt':
            stdbuf = shutil.which('stdbuf')
            if stdbuf:
                return [stdbuf, '-oL', '-eL', exe] + args
    except Exception:
        pass
    return [exe] + args


def build_btc_pattern(address: str, address_type: str) -> Optional[str]:
    """Build BTC minimal network prefix only (random within coin); suffix filtered in stdout."""
    if not address:
        return None

    # Network prefix only
    if address_type == "BTC_P2PKH":
        prefix = "1"
    elif address_type == "BTC_P2SH":
        prefix = "3"
    elif address_type == "BTC_Bech32":
        prefix = "bc1"
    else:
        return None

    return prefix or None


async def generate_btc_with_vpp(address: str, address_type: str) -> Optional[Dict]:
    """Generate BTC using vanitygen-plusplus.
    - P2PKH/P2SH: use GPU suffix mode (-U) with last-4 chars
    - Bech32 (bc1q/p): use proper format (-F p2wpkh/p2tr) and prefix search, filter suffix in stdout
    """
    exes = _find_all_exes()
    if not exes:
        return None

    # Bech32: require format selection and prefix search (GPU suffix not yet supported)
    is_bech32 = (address_type == "BTC_Bech32")
    suffix = address[-4:] if len(address) >= 4 else ""
    for exe in exes:
        if is_bech32:
            # Decide segwit type by hrp+version: bc1p -> p2tr, otherwise p2wpkh
            fmt = "p2tr" if address.startswith("bc1p") else "p2wpkh"
            pattern_prefix = build_btc_pattern(address, address_type)
            if not pattern_prefix:
                continue
            base = ["-q", "-z", "-k", "-F", fmt, pattern_prefix]
        else:
            # P2PKH/P2SH with GPU suffix matching
            fmt_args = []
            if address_type == "BTC_P2SH":
                fmt_args = ["-F", "script"]
            # Use -1 to stop at first match; pass suffix as pattern
            base = ["-q", "-z", "-1"] + fmt_args + ["-U", suffix]
        cmd = _wrap_cmd_for_line_buffering(exe, _maybe_add_device_args(exe, base))
        try:
            _debug_log("exec:", cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            _debug_log("started pid=", proc.pid)
            current_addr = None
            current_priv = None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    rc = await proc.wait()
                    _debug_log("proc exit rc=", rc)
                    break
                text = line.decode(errors="ignore").strip()
                _debug_log("stdout:", text)
                if not text:
                    continue
                # CSV: COIN,PREFIX,ADDRESS,PRIVKEY (when -z)
                if "," in text and (text.startswith("TRX") or text.startswith("BTC") or text.startswith("ETH") or text.startswith("bc")):
                    try:
                        parts = [p.strip() for p in text.split(",")]
                        if len(parts) >= 4:
                            current_addr = parts[2]
                            current_priv = parts[3]
                    except Exception:
                        pass
                elif "Address:" in text:
                    val = text.split("Address:")[-1].strip()
                    current_addr = val.split()[0]
                if "Privkey:" in text:
                    val = text.split("Privkey:")[-1].strip()
                    current_priv = val.split()[0]
                if current_addr and current_priv:
                    if is_bech32:
                        # filter suffix in Python
                        if current_addr.endswith(suffix) if suffix else True:
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            _debug_log("match:", current_addr)
                            return {
                                "address": current_addr,
                                "private_key": current_priv,
                                "type": address_type,
                            }
                        current_addr = None
                        current_priv = None
                    else:
                        # -1 already exited after first match; just return
                        return {
                            "address": current_addr,
                            "private_key": current_priv,
                            "type": address_type,
                        }
        except Exception:
            _debug_log("exec failed")
            continue

    return None


def build_trx_pattern(address: str) -> Optional[str]:
    """Build TRX minimal prefix 'T' only (random TRX); suffix filtered in stdout."""
    if not address or not address.startswith("T") or len(address) < 6:
        return None
    prefix = "T"
    return prefix


async def generate_trx_with_vpp(address: str) -> Optional[Dict]:
    """Generate TRX using GPU prefix; filter last-4 suffix in stdout."""
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_trx_pattern(address)
    if not pattern:
        return None

    suffix = address[-4:] if len(address) >= 4 else ""
    for exe in exes:
        base = ["-q", "-z", "-k", "-C", "TRX", pattern]
        cmd = _wrap_cmd_for_line_buffering(exe, _maybe_add_device_args(exe, base))
        try:
            _debug_log("exec:", cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            _debug_log("started pid=", proc.pid)
            current_addr = None
            current_priv = None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    rc = await proc.wait()
                    _debug_log("proc exit rc=", rc)
                    break
                text = line.decode(errors="ignore").strip()
                _debug_log("stdout:", text)
                if not text:
                    continue
                if "," in text and (text.startswith("TRX") or text.startswith("BTC") or text.startswith("ETH") or text.startswith("bc")):
                    try:
                        parts = [p.strip() for p in text.split(",")]
                        if len(parts) >= 4:
                            current_addr = parts[2]
                            current_priv = parts[3]
                    except Exception:
                        pass
                elif "Address:" in text:
                    val = text.split("Address:")[-1].strip()
                    current_addr = val.split()[0]
                if "Privkey:" in text:
                    val = text.split("Privkey:")[-1].strip()
                    current_priv = val.split()[0]
                if current_addr and current_priv:
                    if current_addr.endswith(suffix) if suffix else True:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        _debug_log("match:", current_addr)
                        return {
                            "address": current_addr,
                            "private_key": current_priv,
                            "type": "TRON",
                        }
                    current_addr = None
                    current_priv = None
        except Exception:
            _debug_log("exec failed")
            continue

    return None


def build_eth_pattern(address: str) -> Optional[str]:
    """Build ETH minimal prefix '0x' only; suffix filtered in stdout."""
    if not address or not address.startswith("0x") or len(address) < 7:
        return None
    prefix = "0x"
    return prefix


async def generate_eth_with_vpp(address: str) -> Optional[Dict]:
    """Generate ETH using GPU suffix (-U) with last-4 hex (case-insensitive)."""
    exes = _find_all_exes()
    if not exes:
        return None

    suffix = address[-4:] if len(address) >= 4 else ""
    for exe in exes:
        # -U suffix matching on GPU; -i case-insensitive; -1 stop at first hit
        base = ["-q", "-z", "-1", "-C", "ETH", "-U", "-i", suffix]
        cmd = _wrap_cmd_for_line_buffering(exe, _maybe_add_device_args(exe, base))
        try:
            _debug_log("exec:", cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(exe)
            )
            _debug_log("started pid=", proc.pid)
            current_addr = None
            current_priv = None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    rc = await proc.wait()
                    _debug_log("proc exit rc=", rc)
                    break
                text = line.decode(errors="ignore").strip()
                _debug_log("stdout:", text)
                if not text:
                    continue
                if "," in text and (text.startswith("TRX") or text.startswith("BTC") or text.startswith("ETH") or text.startswith("bc")):
                    try:
                        parts = [p.strip() for p in text.split(",")]
                        if len(parts) >= 4:
                            current_addr = parts[2]
                            current_priv = parts[3]
                    except Exception:
                        pass
                elif "Address:" in text:
                    val = text.split("Address:")[-1].strip()
                    current_addr = val.split()[0]
                if "Privkey:" in text:
                    val = text.split("Privkey:")[-1].strip()
                    current_priv = val.split()[0]
                if current_addr and current_priv:
                    return {
                        "address": current_addr,
                        "private_key": current_priv,
                        "type": "ETH",
                    }
        except Exception:
            _debug_log("exec failed")
            continue

    return None



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

    # 根据操作系统确定候选文件
    if os.name == 'nt':  # Windows
        exe_candidates = [
            "oclvanitygen++.exe", "oclvanitygen.exe",
            "vanitygen++.exe", "vanitygen.exe",
        ]
    else:  # Linux/Unix
        exe_candidates = [
            "oclvanitygen++", "oclvanitygen",
            "vanitygen++", "vanitygen",
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
    
    # 根据操作系统确定候选文件
    if os.name == 'nt':  # Windows
        # 优先 GPU 版本
        exe_candidates = [
            "oclvanitygen++.exe", "oclvanitygen.exe",
            "vanitygen++.exe", "vanitygen.exe",
        ]
    else:  # Linux/Unix
        # 优先 GPU 版本
        exe_candidates = [
            "oclvanitygen++", "oclvanitygen",
            "vanitygen++", "vanitygen",
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
    """Build BTC pattern: fixed prefix + 4 chars after it"""
    if not address:
        return None

    # Return fixed prefix + 4 chars after it
    if address_type == "BTC_P2PKH" and len(address) >= 5:
        # 1 + 4 chars after it
        return address[:5]
    elif address_type == "BTC_P2SH" and len(address) >= 5:
        # 3 + 4 chars after it
        return address[:5]
    elif address_type == "BTC_Bech32" and len(address) >= 7:
        # bc1 + 4 chars after it
        return address[:7]
    else:
        return None


async def generate_btc_with_vpp(address: str, address_type: str) -> Optional[Dict]:
    """Generate BTC using vanitygen-plusplus with prefix matching (first 4 chars)"""
    exes = _find_all_exes()
    if not exes:
        return None

    pattern_prefix = build_btc_pattern(address, address_type)
    if not pattern_prefix:
        return None
        
    for exe in exes:
        if address_type == "BTC_Bech32":
            # Decide segwit type by hrp+version: bc1p -> p2tr, otherwise p2wpkh
            fmt = "p2tr" if address.startswith("bc1p") else "p2wpkh"
            base = ["-q", "-z", "-k", "-F", fmt, pattern_prefix]
        else:
            # P2PKH/P2SH with prefix matching
            fmt_args = []
            if address_type == "BTC_P2SH":
                fmt_args = ["-F", "script"]
            # Use -1 to stop at first match; use prefix pattern
            base = ["-q", "-z", "-1"] + fmt_args + [pattern_prefix]
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
                    # Direct return since we're using prefix matching
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
    """Build TRX pattern: T + 4 chars after it"""
    if not address or not address.startswith("T") or len(address) < 5:
        return None
    # Return T + 4 chars after it
    return address[:5]


async def generate_trx_with_profanity(address: str) -> Optional[Dict]:
    """Generate TRX using profanity-tron with suffix matching (last 5 chars)"""
    
    # 验证地址必须是34位
    if not address or len(address) != 34:
        _debug_log(f"Invalid address length: {len(address) if address else 0}, must be 34")
        return None
    
    # 获取地址的后5位作为匹配模式
    if not address.startswith('T'):
        _debug_log(f"Invalid TRON address: must start with T")
        return None
    
    suffix_pattern = address[-5:]
    # 构建匹配模式，总共需要20个字符（根据用户说明）
    # 对于20位格式：T + X + 后缀
    # 后5位：T + 15个X + 5位后缀
    # 后6位：T + 14个X + 6位后缀
    x_count = 20 - len(suffix_pattern)  # 直接用20减去后缀长度
    matching_pattern = "T" + "X" * x_count + suffix_pattern
    
    # 构建命令 - 使用全局profanity命令
    cmd = [
        "profanity",
        "--matching", matching_pattern,
        "--suffix-count", "5",
        "--quit-count", "1"
    ]
    
    try:
        _debug_log("exec profanity:", cmd)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        current_addr = None
        current_priv = None
        
        # 也读取stderr，因为profanity的进度信息在stderr
        stderr_lines = []
        
        while True:
            line = await proc.stdout.readline()
            if not line:
                # 检查进程是否结束
                if proc.returncode is not None:
                    break
                # 等待一下
                await asyncio.sleep(0.1)
                continue
                
            text = line.decode(errors="ignore").strip()
            _debug_log("profanity stdout:", text)
            
            # profanity输出格式: 地址 私钥（空格分隔）
            # 示例: TBHHJRWYhxdx7jQjXWyiRizbmquvAAAAAA a559625ec86e4d27dc341362b54f1599ea0ab8b7d5d149286bd98fcbffc5fbc4
            if text and not text.startswith(("Skipping", "Using", "Devices:", "OpenCL:", "Initializing:", "Running", "Before", "GPU-", "Context", "Binary", "Program", "Should be")):
                parts = text.split()
                if len(parts) >= 2:
                    addr_candidate = parts[0]
                    key_candidate = parts[1]
                    
                    # 验证是否为有效的TRON地址和私钥
                    if addr_candidate.startswith("T") and len(addr_candidate) == 34 and len(key_candidate) == 64:
                        if addr_candidate.endswith(suffix_pattern):
                            current_addr = addr_candidate
                            current_priv = key_candidate
                            _debug_log(f"Found matching address: {current_addr}")
                            # 终止进程
                            proc.terminate()
                            break
        
        # 等待进程结束
        await proc.wait()
        
        if current_addr and current_priv:
            return {
                "address": current_addr,
                "private_key": current_priv,
                "type": "TRON",
            }
            
    except Exception as e:
        _debug_log(f"profanity-tron exec failed: {e}")
    
    return None


async def generate_trx_with_vpp(address: str) -> Optional[Dict]:
    """Generate TRX using suffix matching (last 5 chars) with profanity-tron"""
    # 先尝试使用profanity-tron进行后缀匹配
    profanity_result = await generate_trx_with_profanity(address)
    if profanity_result:
        return profanity_result
    
    # 如果profanity-tron失败，回退到vanitygen-plusplus的前缀匹配
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_trx_pattern(address)
    if not pattern:
        return None

    for exe in exes:
        # Use -1 to stop at first match
        base = ["-q", "-z", "-1", "-C", "TRX", pattern]
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
                    # Direct return since we're using prefix matching
                    return {
                        "address": current_addr,
                        "private_key": current_priv,
                        "type": "TRON",
                    }
        except Exception:
            _debug_log("exec failed")
            continue

    return None


def build_eth_pattern(address: str) -> Optional[str]:
    """Build ETH pattern: 0x + 4 hex chars after it"""
    if not address or not address.startswith("0x") or len(address) < 6:
        return None
    # Return 0x + 4 hex chars after it
    return address[:6]


async def generate_eth_with_vpp(address: str) -> Optional[Dict]:
    """Generate ETH using prefix matching (first 4 chars, case-insensitive)"""
    exes = _find_all_exes()
    if not exes:
        return None

    pattern = build_eth_pattern(address)
    if not pattern:
        return None
        
    for exe in exes:
        # -i case-insensitive; -1 stop at first hit
        base = ["-q", "-z", "-1", "-C", "ETH", "-i", pattern]
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



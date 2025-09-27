using System;
using System.Text.RegularExpressions;
using System.Linq;

namespace ClipboardClient
{
    public static class CryptoAddressDetector
    {
        public enum AddressType
        {
            None,
            TRON,
            Bitcoin,
            Ethereum,
            BNB,
            Solana
        }

        public static (AddressType type, bool isValid) DetectAddress(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return (AddressType.None, false);

            text = text.Trim();

            // TRON地址检测
            if (IsTronAddress(text))
                return (AddressType.TRON, true);

            // Bitcoin地址检测
            if (IsBitcoinAddress(text))
                return (AddressType.Bitcoin, true);

            // Ethereum和BNB地址检测（它们使用相同的格式）
            if (IsEthereumAddress(text))
            {
                // 无法仅从地址格式区分ETH和BNB，默认返回ETH
                return (AddressType.Ethereum, true);
            }

            // Solana地址检测
            if (IsSolanaAddress(text))
                return (AddressType.Solana, true);

            return (AddressType.None, false);
        }

        private static bool IsTronAddress(string address)
        {
            // TRON地址：以T开头，34个字符，Base58编码
            if (address.Length != 34 || !address.StartsWith("T"))
                return false;

            return IsBase58(address);
        }

        private static bool IsBitcoinAddress(string address)
        {
            // Bitcoin地址：
            // P2PKH: 以1开头，25-34个字符
            // P2SH: 以3开头，25-34个字符
            // Bech32: 以bc1开头，42-62个字符
            if (address.StartsWith("bc1"))
            {
                return Regex.IsMatch(address, @"^bc1[a-z0-9]{39,59}$");
            }
            else if (address.StartsWith("1") || address.StartsWith("3"))
            {
                return address.Length >= 25 && address.Length <= 34 && IsBase58(address);
            }

            return false;
        }

        private static bool IsEthereumAddress(string address)
        {
            // Ethereum/BNB地址：以0x开头，40个十六进制字符（总共42个字符）
            return Regex.IsMatch(address, @"^0x[a-fA-F0-9]{40}$");
        }

        private static bool IsSolanaAddress(string address)
        {
            // Solana地址：32-44个字符，Base58编码
            if (address.Length < 32 || address.Length > 44)
                return false;

            return IsBase58(address);
        }

        private static bool IsBase58(string text)
        {
            const string base58Chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
            return text.All(c => base58Chars.Contains(c));
        }

        public static string GetAddressTypeName(AddressType type)
        {
            return type switch
            {
                AddressType.TRON => "TRON地址",
                AddressType.Bitcoin => "BTC地址",
                AddressType.Ethereum => "ETH地址",
                AddressType.BNB => "BNB地址",
                AddressType.Solana => "Solana地址",
                _ => "未知"
            };
        }
    }
}

/*
 * 高性能TRON地址GPU生成器 - CUDA C++实现
 * 针对RTX 5070 Ti优化
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
using namespace std;

// 定义uint128_t (Linux/GCC)
#ifdef __GNUC__
    typedef unsigned __int128 uint128_t;
#else
    // Windows/MSVC fallback
    struct uint128_t {
        uint64_t low;
        uint64_t high;
    };
#endif

// secp256k1参数
__constant__ uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

__constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

__constant__ uint64_t SECP256K1_GX[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};

__constant__ uint64_t SECP256K1_GY[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

// Base58字符集
__constant__ char BASE58_ALPHABET[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// 58^i (i=0..10)
__constant__ uint64_t POW58[11] = {
    1ULL,
    58ULL,
    3364ULL,
    195112ULL,
    11316496ULL,
    656356768ULL,
    38068692544ULL,
    2207984167552ULL,
    128063081718016ULL,
    7427658739644928ULL,
    430804206899405824ULL
};

// 256位整数结构
struct uint256_t {
    uint64_t data[4];
};

// 前向声明
__device__ int cmp_256(const uint256_t* a, const uint256_t* b);

// 辅助：从4xuint64装配uint256_t（按现有小端分段顺序）
__device__ __forceinline__ uint256_t make_u256(const uint64_t a[4]) {
    uint256_t r; r.data[0]=a[0]; r.data[1]=a[1]; r.data[2]=a[2]; r.data[3]=a[3]; return r;
}

// 点结构
struct Point {
    uint256_t x;
    uint256_t y;
};

// 基础256位加/减（带正确进位/借位）
__device__ void add_256(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a->data[i] + b->data[i];
        uint64_t c1 = (s < a->data[i]) ? 1 : 0;
        uint64_t s2 = s + carry;
        uint64_t c2 = (s2 < s) ? 1 : 0;
        result->data[i] = s2;
        carry = c1 | c2;
    }
}

__device__ void sub_256(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a->data[i];
        uint64_t bi = b->data[i];
        uint64_t d = ai - bi;
        uint64_t b1 = (ai < bi) ? 1 : 0;
        uint64_t d2 = d - borrow;
        uint64_t b2 = (d < borrow) ? 1 : 0;
        result->data[i] = d2;
        borrow = b1 | b2;
    }
}

// 判断是否为0
__device__ bool is_zero_256(const uint256_t* a) {
    return (a->data[0] | a->data[1] | a->data[2] | a->data[3]) == 0ULL;
}

// 设定为常数 0 或 1
__device__ void set_zero_256(uint256_t* a) { a->data[0]=a->data[1]=a->data[2]=a->data[3]=0ULL; }
__device__ void set_one_256(uint256_t* a) { a->data[0]=1ULL; a->data[1]=a->data[2]=a->data[3]=0ULL; }

// 有限域p上的加减乘（secp256k1）
__device__ void add_mod_p(uint256_t* r, const uint256_t* x, const uint256_t* y) {
    uint256_t t;
    add_256(&t, x, y);
    uint256_t P = make_u256(SECP256K1_P);
    if (cmp_256(&t, &P) >= 0) {
        sub_256(&t, &t, &P);
    }
    *r = t;
}

__device__ void sub_mod_p(uint256_t* r, const uint256_t* x, const uint256_t* y) {
    uint256_t P = make_u256(SECP256K1_P);
    if (cmp_256(x, y) >= 0) {
        sub_256(r, x, y);
    } else {
        uint256_t t; add_256(&t, x, &P); sub_256(r, &t, y);
    }
}

// n域加一：r = (r + 1) mod n
__device__ void add_one_mod_n(uint256_t* r) {
    uint256_t N = make_u256(SECP256K1_N);
    uint64_t carry = 1ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t s = r->data[i] + carry;
        carry = (s < r->data[i]) ? 1ULL : 0ULL;
        r->data[i] = s;
        if (!carry) break;
    }
    if (cmp_256(r, &N) >= 0) {
        sub_256(r, r, &N);
    }
}

// 256x256 -> 512 宽乘
__device__ void mul_wide_256(uint64_t out[8], const uint256_t* a, const uint256_t* b) {
    for (int i = 0; i < 8; i++) out[i] = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a->data[i];
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            uint64_t bj = b->data[j];
            uint64_t lo = ai * bj;
            uint64_t hi = __umul64hi(ai, bj);

            // out[i+j] += lo + carry
            uint64_t s = out[i + j] + lo;
            uint64_t c0 = (s < out[i + j]) ? 1 : 0;
            uint64_t s2 = s + carry;
            uint64_t c1 = (s2 < s) ? 1 : 0;
            out[i + j] = s2;

            carry = hi + c0 + c1;
        }
        // 传播剩余进位
        int k = i + 4;
        while (carry != 0ULL) {
            uint64_t s = out[k] + carry;
            uint64_t c = (s < carry) ? 1 : 0;
            out[k] = s;
            carry = c;
            k++;
        }
    }
}

// 针对 p=2^256-2^32-977 的快速归约
__device__ void mod_p(uint256_t* r, const uint64_t x[8]) {
    // 初步折叠：t = L + (H<<32) + 977*H
    uint64_t T[5]; for (int i=0;i<5;i++) T[i]=0ULL;
    // L 部分
    for (int i=0;i<4;i++) T[i] = x[i];

    // 加 (H << 32)
    uint64_t carry = 0ULL;
    for (int i=0;i<4;i++) {
        uint64_t low = x[4 + i] << 32;
        uint64_t high = x[4 + i] >> 32;
        // T[i] += low
        uint64_t s = T[i] + low;
        uint64_t c0 = (s < T[i]) ? 1 : 0;
        T[i] = s;
        // T[i+1] += high + c0
        uint64_t s2 = T[i+1] + high + c0;
        uint64_t c1 = (s2 < T[i+1]) ? 1 : 0;
        if (high + c0 > s2) c1 = 1; // 更稳健（避免优化器合并）
        T[i+1] = s2;
        // 进位继续向更高位传播（很少发生）
        int k = i + 2;
        uint64_t c = c1 ? 1ULL : 0ULL;
        while (c && k < 5) { uint64_t s3 = T[k] + c; uint64_t c2 = (s3 < T[k]) ? 1 : 0; T[k]=s3; c = c2; k++; }
    }

    // 加 977*H
    for (int i=0;i<4;i++) {
        uint64_t hk = x[4+i];
        uint64_t low = hk * 977ULL;
        uint64_t high = __umul64hi(hk, 977ULL);
        // T[i] += low
        uint64_t s = T[i] + low;
        uint64_t c0 = (s < T[i]) ? 1 : 0;
        T[i] = s;
        // T[i+1] += high + c0
        uint64_t s2 = T[i+1] + high + c0;
        uint64_t c1 = (s2 < T[i+1]) ? 1 : 0; if (high + c0 > s2) c1 = 1;
        T[i+1] = s2;
        int k = i + 2; uint64_t c = c1 ? 1ULL : 0ULL;
        while (c && k < 5) { uint64_t s3 = T[k] + c; uint64_t c2 = (s3 < T[k]) ? 1 : 0; T[k]=s3; c = c2; k++; }
    }

    // 第二次折叠：仅用 H2 = T[4]
    uint64_t V[4]; for (int i=0;i<4;i++) V[i]=T[i];
    uint64_t H2 = T[4];
    if (H2) {
        // 加 (H2 << 32)
        uint64_t low = H2 << 32;
        uint64_t high = H2 >> 32;
        uint64_t s = V[0] + low; uint64_t c0 = (s < V[0]) ? 1 : 0; V[0]=s;
        uint64_t s2 = V[1] + high + c0; uint64_t c1 = (s2 < V[1]) ? 1 : 0; if (high + c0 > s2) c1 = 1; V[1]=s2;
        int k=2; uint64_t c=c1?1ULL:0ULL; while (c && k<4){ uint64_t s3=V[k]+c; uint64_t c2=(s3<V[k])?1:0; V[k]=s3; c=c2; k++; }

        // 加 977*H2
        uint64_t low2 = H2 * 977ULL;
        uint64_t high2 = __umul64hi(H2, 977ULL);
        s = V[0] + low2; c0 = (s < V[0]) ? 1 : 0; V[0]=s;
        s2 = V[1] + high2 + c0; c1 = (s2 < V[1]) ? 1 : 0; if (high2 + c0 > s2) c1 = 1; V[1]=s2;
        k=2; c=c1?1ULL:0ULL; while (c && k<4){ uint64_t s3=V[k]+c; uint64_t c2=(s3<V[k])?1:0; V[k]=s3; c=c2; k++; }
    }

    uint256_t tmp; tmp.data[0]=V[0]; tmp.data[1]=V[1]; tmp.data[2]=V[2]; tmp.data[3]=V[3];
    uint256_t P = make_u256(SECP256K1_P);
    // 最终条件减 p（最多两次）
    if (cmp_256(&tmp, &P) >= 0) { sub_256(&tmp, &tmp, &P); }
    if (cmp_256(&tmp, &P) >= 0) { sub_256(&tmp, &tmp, &P); }
    *r = tmp;
}

__device__ void mul_mod_p(uint256_t* r, const uint256_t* a, const uint256_t* b) {
    uint64_t w[8];
    mul_wide_256(w, a, b);
    mod_p(r, w);
}

__device__ void sqr_mod_p(uint256_t* r, const uint256_t* a) {
    mul_mod_p(r, a, a);
}

__device__ void mul_256(uint256_t* result, const uint256_t* a, const uint256_t* b) {
    // 完整的256位乘法实现
    uint64_t temp[8] = {0};
    
    // 逐位乘法
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // 将64位乘法拆分为32位以避免溢出
            uint64_t a_lo = a->data[i] & 0xFFFFFFFFULL;
            uint64_t a_hi = a->data[i] >> 32;
            uint64_t b_lo = b->data[j] & 0xFFFFFFFFULL;
            uint64_t b_hi = b->data[j] >> 32;
            
            uint64_t p0 = a_lo * b_lo;
            uint64_t p1 = a_hi * b_lo;
            uint64_t p2 = a_lo * b_hi;
            uint64_t p3 = a_hi * b_hi;
            
            uint64_t cy = p0 >> 32;
            p1 += cy;
            cy = p1 >> 32;
            p1 = (p1 & 0xFFFFFFFFULL) + p2;
            cy += p1 >> 32;
            p3 += cy;
            
            uint64_t lo = (p0 & 0xFFFFFFFFULL) | ((p1 & 0xFFFFFFFFULL) << 32);
            uint64_t hi = p3;
            
            // 累加到结果
            if (i + j < 8) {
                uint64_t sum = temp[i + j] + lo;
                uint64_t carry = (sum < temp[i + j]) ? 1 : 0;
                temp[i + j] = sum;
                
                if (i + j + 1 < 8) {
                    temp[i + j + 1] += hi + carry;
                }
            }
        }
    }
    
    // 取低256位
    for (int i = 0; i < 4; i++) {
        result->data[i] = temp[i];
    }
}

// 保留但不再用于场p：通用模（仅限小数值，避免误用）
__device__ void mod_256(uint256_t* result, const uint256_t* a, const uint256_t* m) {
    uint256_t temp = *a;
    while (cmp_256(&temp, m) >= 0) { sub_256(&temp, &temp, m); }
    *result = temp;
}

// 比较两个256位数
__device__ int cmp_256(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

// 模逆：使用 a^(p-2) mod p（依赖正确的 mul_mod_p）
__device__ void mod_inverse_p(uint256_t* result, const uint256_t* a) {
    // exp = p-2
    uint256_t exp; exp.data[0]=0xFFFFFFFEFFFFFC2DULL; exp.data[1]=0xFFFFFFFFFFFFFFFFULL; exp.data[2]=0xFFFFFFFFFFFFFFFFULL; exp.data[3]=0xFFFFFFFFFFFFFFFFULL;
    uint256_t base = *a;
    uint256_t res; set_one_256(&res);

    for (int i = 0; i < 256; i++) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        if ((exp.data[word_idx] >> bit_idx) & 1ULL) {
            uint256_t t;
            mul_mod_p(&t, &res, &base);
            res = t;
        }
        if (i < 255) {
            uint256_t t2; mul_mod_p(&t2, &base, &base); base = t2;
        }
    }
    *result = res;
}

// Jacobian 坐标
struct JPoint { uint256_t X; uint256_t Y; uint256_t Z; };

__device__ bool is_infinity(const JPoint* P) { return is_zero_256(&P->Z); }

__device__ void set_infinity(JPoint* R) { set_zero_256(&R->X); set_zero_256(&R->Y); set_zero_256(&R->Z); }

__device__ void point_double_jacobian(JPoint* R, const JPoint* P) {
    if (is_infinity(P)) { *R = *P; return; }
    uint256_t XX, YY, YYYY, ZZ, S, M;
    // XX = X1^2
    sqr_mod_p(&XX, &P->X);
    // YY = Y1^2
    sqr_mod_p(&YY, &P->Y);
    // YYYY = YY^2
    sqr_mod_p(&YYYY, &YY);
    // ZZ = Z1^2
    sqr_mod_p(&ZZ, &P->Z);
    // S = 4 * X1 * YY
    uint256_t t1; mul_mod_p(&t1, &P->X, &YY); add_mod_p(&S, &t1, &t1); add_mod_p(&S, &S, &S); // *4
    // M = 3 * XX
    add_mod_p(&M, &XX, &XX); add_mod_p(&M, &M, &XX);
    // T = M^2 - 2*S
    uint256_t M2; sqr_mod_p(&M2, &M);
    uint256_t twoS; add_mod_p(&twoS, &S, &S);
    sub_mod_p(&R->X, &M2, &twoS);
    // Y3 = M*(S - X3) - 8*YYYY
    uint256_t S_minus_X3; sub_mod_p(&S_minus_X3, &S, &R->X);
    uint256_t M_mul; mul_mod_p(&M_mul, &M, &S_minus_X3);
    uint256_t eightYYYY; add_mod_p(&eightYYYY, &YYYY, &YYYY); // 2*YYYY
    add_mod_p(&eightYYYY, &eightYYYY, &eightYYYY); // 4*
    add_mod_p(&eightYYYY, &eightYYYY, &eightYYYY); // 8*
    sub_mod_p(&R->Y, &M_mul, &eightYYYY);
    // Z3 = 2*Y1*Z1
    uint256_t YZ; mul_mod_p(&YZ, &P->Y, &P->Z);
    add_mod_p(&R->Z, &YZ, &YZ);
}

__device__ void point_add_jacobian(JPoint* R, const JPoint* P, const JPoint* Q) {
    if (is_infinity(P)) { *R = *Q; return; }
    if (is_infinity(Q)) { *R = *P; return; }
    uint256_t Z1Z1, Z2Z2, U1, U2, S1, S2;
    sqr_mod_p(&Z1Z1, &P->Z);
    sqr_mod_p(&Z2Z2, &Q->Z);
    // U1 = X1*Z2^2 ; U2 = X2*Z1^2
    mul_mod_p(&U1, &P->X, &Z2Z2);
    mul_mod_p(&U2, &Q->X, &Z1Z1);
    // S1 = Y1*Z2^3 ; S2 = Y2*Z1^3
    uint256_t Z2Z3, Z1Z3;
    mul_mod_p(&Z2Z3, &Z2Z2, &Q->Z);
    mul_mod_p(&Z1Z3, &Z1Z1, &P->Z);
    mul_mod_p(&S1, &P->Y, &Z2Z3);
    mul_mod_p(&S2, &Q->Y, &Z1Z3);

    if (cmp_256(&U1, &U2) == 0) {
        // 同x
        if (cmp_256(&S1, &S2) != 0) { set_infinity(R); return; }
        point_double_jacobian(R, P); return;
    }

    uint256_t H, Rr; sub_mod_p(&H, &U2, &U1); sub_mod_p(&Rr, &S2, &S1);
    uint256_t H2; sqr_mod_p(&H2, &H);
    uint256_t H3; mul_mod_p(&H3, &H, &H2);
    uint256_t U1H2; mul_mod_p(&U1H2, &U1, &H2);
    // X3 = Rr^2 - H3 - 2*U1H2
    uint256_t Rr2; sqr_mod_p(&Rr2, &Rr);
    uint256_t twoU1H2; add_mod_p(&twoU1H2, &U1H2, &U1H2);
    uint256_t tmp; sub_mod_p(&tmp, &Rr2, &H3); sub_mod_p(&R->X, &tmp, &twoU1H2);
    // Y3 = Rr*(U1H2 - X3) - S1*H3
    uint256_t U1H2_minus_X3; sub_mod_p(&U1H2_minus_X3, &U1H2, &R->X);
    uint256_t Rmul; mul_mod_p(&Rmul, &Rr, &U1H2_minus_X3);
    uint256_t S1H3; mul_mod_p(&S1H3, &S1, &H3);
    sub_mod_p(&R->Y, &Rmul, &S1H3);
    // Z3 = H * Z1 * Z2
    uint256_t Z1Z2; mul_mod_p(&Z1Z2, &P->Z, &Q->Z);
    mul_mod_p(&R->Z, &Z1Z2, &H);
}

__device__ void scalar_mul(JPoint* R, const uint256_t* k) {
    set_infinity(R);
    // 基点（Jacobian）
    JPoint G; G.X = make_u256(SECP256K1_GX); G.Y = make_u256(SECP256K1_GY); set_one_256(&G.Z);

    for (int bit = 255; bit >= 0; bit--) {
        // R = 2R
        JPoint T; point_double_jacobian(&T, R); *R = T;
        int w = bit / 64; int b = bit % 64;
        if ((k->data[w] >> b) & 1ULL) {
            JPoint U; point_add_jacobian(&U, R, &G); *R = U;
        }
    }
}

__device__ void jacobian_to_uncompressed65(const JPoint* P, uint8_t out65[65]) {
    // 处理无穷远点（不应发生在有效私钥）
    if (is_infinity(P)) { out65[0]=0x04; for(int i=1;i<65;i++) out65[i]=0; return; }
    uint256_t Zinv, Zinv2, Zinv3, X, Y;
    mod_inverse_p(&Zinv, &P->Z);
    sqr_mod_p(&Zinv2, &Zinv);
    mul_mod_p(&Zinv3, &Zinv2, &Zinv);
    mul_mod_p(&X, &P->X, &Zinv2);
    mul_mod_p(&Y, &P->Y, &Zinv3);
    out65[0] = 0x04;
    // 大端序序列化
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            out65[1 + i * 8 + j] = (X.data[3 - i] >> (56 - j * 8)) & 0xFF;
            out65[33 + i * 8 + j] = (Y.data[3 - i] >> (56 - j * 8)) & 0xFF;
        }
    }
}

// 旧仿射标量乘不再使用

// Keccak-256轮常数
__constant__ uint64_t KECCAK_ROUND_CONSTANTS[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak-256旋转偏移量
__constant__ int KECCAK_ROTATION_OFFSETS[25] = {
    // row-major (index = x + 5*y), offsets defined on source (x,y)
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Reference Keccak (XKCP style): theta, rho, pi, chi, iota
__device__ __forceinline__ uint64_t ROTL64(uint64_t x, int n) {
    n &= 63;
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_theta_ref(uint64_t A[25]) {
    uint64_t C[5], D[5];
    for (int x = 0; x < 5; x++) {
        C[x] = A[x] ^ A[x+5] ^ A[x+10] ^ A[x+15] ^ A[x+20];
    }
    for (int x = 0; x < 5; x++) {
        D[x] = C[(x+4)%5] ^ ROTL64(C[(x+1)%5], 1);
    }
    for (int i = 0; i < 25; i++) {
        A[i] ^= D[i % 5];
    }
}

__device__ void keccak_rho_pi_ref(uint64_t A[25]) {
    // rotation offsets r and pi mapping
    const int r[25] = {
         0,  36,   3, 105, 210,
         1,  44,  10,  45,  66,
        62,   6,  43,  15, 253,
        28,  55, 153,  21, 120,
        27,  20,  39,   8,  14
    };
    const int pi[25] = {
         0,  6, 12, 18, 24,
         3,  9, 10, 16, 22,
         1,  7, 13, 19, 20,
         4,  5, 11, 17, 23,
         2,  8, 14, 15, 21
    };
    uint64_t B[25];
    for (int i = 0; i < 25; i++) {
        B[pi[i]] = ROTL64(A[i], r[i]);
    }
    for (int i = 0; i < 25; i++) {
        A[i] = B[i];
    }
}

__device__ void keccak_chi_ref(uint64_t A[25]) {
    for (int y = 0; y < 5; y++) {
        uint64_t a0 = A[5*y+0], a1 = A[5*y+1], a2 = A[5*y+2], a3 = A[5*y+3], a4 = A[5*y+4];
        A[5*y+0] = a0 ^ ((~a1) & a2);
        A[5*y+1] = a1 ^ ((~a2) & a3);
        A[5*y+2] = a2 ^ ((~a3) & a4);
        A[5*y+3] = a3 ^ ((~a4) & a0);
        A[5*y+4] = a4 ^ ((~a0) & a1);
    }
}

__device__ void keccak_iota_ref(uint64_t A[25], int round) {
    A[0] ^= KECCAK_ROUND_CONSTANTS[round];
}

__device__ void keccak_f_ref(uint64_t A[25]) {
    for (int round = 0; round < 24; round++) {
        keccak_theta_ref(A);
        keccak_rho_pi_ref(A);
        keccak_chi_ref(A);
        keccak_iota_ref(A, round);
    }
}

// Keccak-256 theta函数
__device__ void keccak_theta(uint64_t state[25]) {
    uint64_t C[5], D[5];
    
    for (int i = 0; i < 5; i++) {
        C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    }
    
    for (int i = 0; i < 5; i++) {
        D[i] = C[(i + 4) % 5] ^ ((C[(i + 1) % 5] << 1) | (C[(i + 1) % 5] >> 63));
    }
    
    for (int i = 0; i < 25; i++) {
        state[i] ^= D[i % 5];
    }
}

// Keccak-256 rho和pi函数
__device__ void keccak_rho_pi(uint64_t state[25]) {
    // Correct combined rho+pi: rotation offset is defined on SOURCE (x,y), not destination
    uint64_t current = state[1];
    int x = 1, y = 0;
    for (int t = 0; t < 24; t++) {
        int X = y;
        int Y = (2 * x + 3 * y) % 5;   // π mapping
        int dst = X + 5 * Y;           // destination index
        int src = x + 5 * y;           // source index (for rho offset)
        uint64_t temp = state[dst];
        int r = KECCAK_ROTATION_OFFSETS[src];
        state[dst] = (current << r) | (current >> (64 - r));
        current = temp;
        x = X; y = Y;
    }
}

// Keccak-256 chi函数
__device__ void keccak_chi(uint64_t state[25]) {
    uint64_t temp[5];
    
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            temp[x] = state[x + 5 * y];
        }
        for (int x = 0; x < 5; x++) {
            state[x + 5 * y] = temp[x] ^ ((~temp[(x + 1) % 5]) & temp[(x + 2) % 5]);
        }
    }
}

// Keccak-256 iota函数
__device__ void keccak_iota(uint64_t state[25], int round) {
    state[0] ^= KECCAK_ROUND_CONSTANTS[round];
}

// Keccak-f[1600]函数
__device__ void keccak_f(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        keccak_theta(state);
        keccak_rho_pi(state);
        keccak_chi(state);
        keccak_iota(state, round);
    }
}

// Keccak 小端装载/存储
__device__ __forceinline__ uint64_t load_le64(const uint8_t* p) {
    return ((uint64_t)p[0])       | ((uint64_t)p[1] << 8 ) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// 完整的Keccak-256实现
__device__ void keccak256(uint8_t* output, const uint8_t* input, size_t len) {
    uint64_t state[25] = {0};
    const size_t rate = 136; // 1088 bits / 8 = 136 bytes
    size_t blockSize = 0;
    
    // 吸收阶段（逐字节异或，保证正确性）
    while (len > 0) {
        size_t toAbsorb = (len < rate - blockSize) ? len : rate - blockSize;
        for (size_t i = 0; i < toAbsorb; i++) {
            ((uint8_t*)state)[blockSize + i] ^= input[i];
        }
        blockSize += toAbsorb;
        input += toAbsorb;
        len -= toAbsorb;
        if (blockSize == rate) {
            keccak_f(state);
            blockSize = 0;
        }
    }
    
    // 填充
    ((uint8_t*)state)[blockSize] ^= 0x01;
    ((uint8_t*)state)[rate - 1] ^= 0x80;
    keccak_f(state);
    
    // 挤压阶段 - 输出32字节（按小端顺序读出）
    memcpy(output, (uint8_t*)state, 32);
}

// SHA-256常数
__constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256初始哈希值
__constant__ uint32_t SHA256_H[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256辅助函数
__device__ uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sha256_sig0(uint32_t x) {
    return sha256_rotr(x, 2) ^ sha256_rotr(x, 13) ^ sha256_rotr(x, 22);
}

__device__ uint32_t sha256_sig1(uint32_t x) {
    return sha256_rotr(x, 6) ^ sha256_rotr(x, 11) ^ sha256_rotr(x, 25);
}

__device__ uint32_t sha256_gamma0(uint32_t x) {
    return sha256_rotr(x, 7) ^ sha256_rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t sha256_gamma1(uint32_t x) {
    return sha256_rotr(x, 17) ^ sha256_rotr(x, 19) ^ (x >> 10);
}

// 完整的SHA-256实现
__device__ void sha256(uint8_t* output, const uint8_t* input, size_t len) {
    uint32_t h[8];
    for (int i = 0; i < 8; i++) h[i] = SHA256_H[i];
    
    // 处理完整的512位块
    size_t processed = 0;
    while (processed + 64 <= len) {
        uint32_t w[64];
        
        // 准备消息调度
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)input[processed + i*4] << 24) |
                   ((uint32_t)input[processed + i*4 + 1] << 16) |
                   ((uint32_t)input[processed + i*4 + 2] << 8) |
                   ((uint32_t)input[processed + i*4 + 3]);
        }
        
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
        }
        
        // 工作变量
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_temp = h[7];
        
        // 主循环
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_temp + sha256_sig1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
            uint32_t t2 = sha256_sig0(a) + sha256_maj(a, b, c);
            
            h_temp = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        
        // 更新哈希值
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_temp;
        
        processed += 64;
    }
    
    // 处理剩余的字节（完整填充，支持 1 或 2 个末块）
    uint8_t padded[128] = {0};
    size_t remaining = len - processed;
    memcpy(padded, input + processed, remaining);
    padded[remaining] = 0x80;
    uint64_t bitLen = (uint64_t)len * 8ULL;

    if (remaining <= 55) {
        for (int i = 0; i < 8; i++) {
            padded[56 + i] = (uint8_t)((bitLen >> ((7 - i) * 8)) & 0xFFULL);
        }
        // 处理 1 个 64B 末块
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)padded[i*4] << 24) |
                   ((uint32_t)padded[i*4 + 1] << 16) |
                   ((uint32_t)padded[i*4 + 2] << 8) |
                   ((uint32_t)padded[i*4 + 3]);
        }
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e2 = h[4], f2 = h[5], g2 = h[6], h_temp2 = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_temp2 + sha256_sig1(e2) + sha256_ch(e2, f2, g2) + SHA256_K[i] + w[i];
            uint32_t t2 = sha256_sig0(a) + sha256_maj(a, b, c);
            h_temp2 = g2; g2 = f2; f2 = e2; e2 = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e2; h[5] += f2; h[6] += g2; h[7] += h_temp2;
    } else {
        // 两个末块
        for (int i = 0; i < 8; i++) {
            padded[120 + i] = (uint8_t)((bitLen >> ((7 - i) * 8)) & 0xFFULL);
        }
        // 处理 padded[0..63]
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)padded[i*4] << 24) |
                   ((uint32_t)padded[i*4 + 1] << 16) |
                   ((uint32_t)padded[i*4 + 2] << 8) |
                   ((uint32_t)padded[i*4 + 3]);
        }
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e2 = h[4], f2 = h[5], g2 = h[6], h_temp2 = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_temp2 + sha256_sig1(e2) + sha256_ch(e2, f2, g2) + SHA256_K[i] + w[i];
            uint32_t t2 = sha256_sig0(a) + sha256_maj(a, b, c);
            h_temp2 = g2; g2 = f2; f2 = e2; e2 = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e2; h[5] += f2; h[6] += g2; h[7] += h_temp2;

        // 处理 padded[64..127]
        for (int i = 0; i < 16; i++) {
            int base = 64 + i*4;
            w[i] = ((uint32_t)padded[base] << 24) |
                   ((uint32_t)padded[base + 1] << 16) |
                   ((uint32_t)padded[base + 2] << 8) |
                   ((uint32_t)padded[base + 3]);
        }
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
        }
        a = h[0]; b = h[1]; c = h[2]; d = h[3];
        e2 = h[4]; f2 = h[5]; g2 = h[6]; h_temp2 = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_temp2 + sha256_sig1(e2) + sha256_ch(e2, f2, g2) + SHA256_K[i] + w[i];
            uint32_t t2 = sha256_sig0(a) + sha256_maj(a, b, c);
            h_temp2 = g2; g2 = f2; f2 = e2; e2 = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e2; h[5] += f2; h[6] += g2; h[7] += h_temp2;
    }
    
    // 输出结果
    for (int i = 0; i < 8; i++) {
        output[i*4] = (h[i] >> 24) & 0xFF;
        output[i*4 + 1] = (h[i] >> 16) & 0xFF;
        output[i*4 + 2] = (h[i] >> 8) & 0xFF;
        output[i*4 + 3] = h[i] & 0xFF;
    }
}

// Base58编码
__device__ void base58_encode(char* output, const uint8_t* input, size_t len) {
    // 大数除法实现Base58编码
    uint8_t temp[256];
    memcpy(temp, input, len);
    
    int out_idx = 0;
    
    // 处理前导零
    int zeros = 0;
    for (int i = 0; i < len && input[i] == 0; i++) {
        zeros++;
    }
    
    // 转换为Base58
    while (len > 0) {
        int remainder = 0;
        for (int i = 0; i < len; i++) {
            int value = remainder * 256 + temp[i];
            temp[i] = value / 58;
            remainder = value % 58;
        }
        output[out_idx++] = BASE58_ALPHABET[remainder];
        
        // 移除前导零
        while (len > 0 && temp[0] == 0) {
            for (int i = 0; i < len - 1; i++) {
                temp[i] = temp[i + 1];
            }
            len--;
        }
    }
    
    // 添加前导1
    for (int i = 0; i < zeros; i++) {
        output[out_idx++] = '1';
    }
    
    // 反转结果
    for (int i = 0; i < out_idx / 2; i++) {
        char t = output[i];
        output[i] = output[out_idx - 1 - i];
        output[out_idx - 1 - i] = t;
    }
    
    output[out_idx] = '\0';
}

// 验证Base58地址格式
__device__ bool is_valid_base58_char(char c) {
    // Base58字符集：123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
    return (c >= '1' && c <= '9') || 
           (c >= 'A' && c <= 'H') || (c >= 'J' && c <= 'N') || (c >= 'P' && c <= 'Z') ||
           (c >= 'a' && c <= 'k') || (c >= 'm' && c <= 'z');
}

// ---- 设备端工具函数：strlen ----
__device__ __forceinline__ int d_strlen(const char* s) {
    int n = 0;
    while (s[n] != '\0') ++n;
    return n;
}

// Base58 字符到索引（-1 无效）
__device__ __forceinline__ int base58_index(char c) {
    if (c >= '1' && c <= '9') return c - '1';
    if (c >= 'A' && c <= 'H') return 9 + (c - 'A');
    if (c >= 'J' && c <= 'N') return 17 + (c - 'J');
    if (c >= 'P' && c <= 'Z') return 22 + (c - 'P');
    if (c >= 'a' && c <= 'k') return 33 + (c - 'a');
    if (c >= 'm' && c <= 'z') return 44 + (c - 'm');
    return -1;
}

// 计算后缀字符串的数值及长度（Base58），返回值放入 out_val
__device__ void suffix_value(const char* suffix, uint64_t* out_val, int* out_len) {
    int len = d_strlen(suffix);
    if (len <= 0 || len > 10) { *out_val = 0ULL; *out_len = 0; return; }
    uint64_t v = 0ULL;
    for (int i = 0; i < len; ++i) {
        int digit = base58_index(suffix[len - 1 - i]);
        if (digit < 0) { *out_val = 0ULL; *out_len = 0; return; }
        v += (uint64_t)digit * POW58[i];
    }
    *out_val = v;
    *out_len = len;
}

// 计算 25 字节 payload 对 58^len 的模
__device__ uint64_t mod_58pow_25(const uint8_t payload[25], int len) {
    if (len <= 0 || len > 10) return 0ULL;

    // Fast and robust path for last <=5 Base58 digits (58^5 < 2^31)
    if (len <= 5) {
        uint32_t M = (uint32_t)POW58[len];
        uint64_t r = 0ULL; // use 64-bit accumulator to avoid r*256 overflow
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            r = ((r * 256u) + (uint32_t)payload[i]) % M;
        }
        return (uint32_t)r;
    }

    // Fallback for 6..10 without __int128: use eight doublings modulo M
    const uint64_t M = POW58[len];
    uint64_t r = 0ULL;
    #pragma unroll
    for (int i = 0; i < 25; ++i) {
        // r = (r * 256) % M via eight safe doublings
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            r <<= 1;
            if (r >= M) r -= M;
        }
        r += (uint64_t)payload[i];
        if (r >= M) r -= M;
    }
    return r;
}

// 序列化仿射 X,Y 为 64 字节（大端）
__device__ void serialize_xy64(const uint256_t* X, const uint256_t* Y, uint8_t out64[64]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            out64[i * 8 + j] = (X->data[3 - i] >> (56 - j * 8)) & 0xFF;
            out64[32 + i * 8 + j] = (Y->data[3 - i] >> (56 - j * 8)) & 0xFF;
        }
    }
}

// 由仿射 X,Y 构建 25 字节 Base58Check 输入（含校验）
__device__ void full_addr_from_xy(const uint256_t* X, const uint256_t* Y, uint8_t out25[25]) {
    uint8_t pub64[64]; serialize_xy64(X, Y, pub64);
    uint8_t hash[32]; keccak256(hash, pub64, 64);
    uint8_t addr21[21]; addr21[0] = 0x41; memcpy(addr21 + 1, hash + 12, 20);
    uint8_t sha1[32], sha2[32];
    sha256(sha1, addr21, 21); sha256(sha2, sha1, 32);
    memcpy(out25, addr21, 21); memcpy(out25 + 21, sha2, 4);
}

// 生成单个TRON地址
__device__ void generate_tron_address(const uint256_t* private_key, char* address) {
    // 调试：打印私钥（仅在第一个线程）
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // printf("Private key: %016llx%016llx%016llx%016llx\n", 
        //        private_key->data[3], private_key->data[2], 
        //        private_key->data[1], private_key->data[0]);
    }
    
    // 1. 计算公钥（Jacobian）
    JPoint R; scalar_mul(&R, private_key);
    // 2. 仿射化并导出非压缩65字节公钥（0x04||X||Y）
    uint8_t pubkey65[65]; jacobian_to_uncompressed65(&R, pubkey65);
    
    // 3. Keccak-256哈希
    uint8_t hash[32];
    // 以太坊/Tron对未压缩公钥使用去掉前缀的 (X||Y) 共64字节
    keccak256(hash, pubkey65 + 1, 64);
    
    // 4. 构建地址（0x41前缀 + 后20字节）
    uint8_t addr_bytes[21];
    addr_bytes[0] = 0x41;
    memcpy(addr_bytes + 1, hash + 12, 20);
    
    // 5. 双SHA256校验和
    uint8_t sha1[32], sha2[32];
    sha256(sha1, addr_bytes, 21);
    sha256(sha2, sha1, 32);
    
    // 6. 添加校验和
    uint8_t full_addr[25];
    memcpy(full_addr, addr_bytes, 21);
    memcpy(full_addr + 21, sha2, 4);
    
    // 7. Base58编码（保留设备端实现，但建议主机端做）
    base58_encode(address, full_addr, 25);
    
    // 可选：仅调试时做轻量校验，避免未使用告警
    #ifdef DEBUG_ADDR_CHECK
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bool ok = (address[0] == 'T') && (d_strlen(address) == 34);
        for (int i = 0; ok && address[i] != '\0'; ++i) {
            if (!is_valid_base58_char(address[i])) ok = false;
        }
        (void)ok;
    }
    #endif
}

// 直接从 Jacobian 点生成 TRON 地址（避免重复标量乘）
__device__ void address_from_point(const JPoint* P_jacobian, char* address) {
    // 仿射化并导出非压缩65字节公钥（0x04||X||Y）
    uint8_t pubkey65[65]; jacobian_to_uncompressed65(P_jacobian, pubkey65);
    // Keccak-256
    uint8_t hash[32];
    keccak256(hash, pubkey65 + 1, 64);
    // 0x41 前缀 + 后20字节
    uint8_t addr_bytes[21];
    addr_bytes[0] = 0x41;
    memcpy(addr_bytes + 1, hash + 12, 20);
    // 双 SHA256 校验和
    uint8_t sha1[32], sha2[32];
    sha256(sha1, addr_bytes, 21);
    sha256(sha2, sha1, 32);
    // 25B 拼接
    uint8_t full_addr[25];
    memcpy(full_addr, addr_bytes, 21);
    memcpy(full_addr + 21, sha2, 4);
    // Base58 编码
    base58_encode(address, full_addr, 25);
}

// 模式匹配
__device__ bool match_pattern(const char* address, const char* prefix, const char* suffix) {
    // 新规则：仅匹配后缀（后N位），忽略前缀
    if (address[0] != 'T' || d_strlen(address) != 34) return false;
    if (suffix[0] == '\0') return true;
    int addr_len = d_strlen(address);
    int suffix_len = d_strlen(suffix);
    for (int i = 0; i < suffix_len; i++) {
        if (address[addr_len - suffix_len + i] != suffix[i]) return false;
    }
    return true;
}

// GPU批量生成内核
// ---- 调试统计与采样（调试完可移除） ----
__device__ unsigned long long g_total = 0ULL;
__device__ unsigned long long g_valid = 0ULL;
__device__ unsigned long long g_checksum_total = 0ULL;
__device__ unsigned long long g_checksum_ok = 0ULL;
__device__ uint8_t g_sample_full25[25];
// 逐位尾缀命中统计（尾 1..5 位）
__device__ unsigned long long g_tail1 = 0ULL, g_tail2 = 0ULL, g_tail3 = 0ULL, g_tail4 = 0ULL, g_tail5 = 0ULL;
__device__ unsigned long long g_tailN_total = 0ULL;

__global__ void generate_batch(
    uint64_t seed,
    char* addresses,
    uint256_t* private_keys,
    bool* matches,
    uint64_t target_mod,
    int suffix_len,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    // 不再依赖设备端字符串后缀，改用 host 侧传入的数值+长度
    
    // 生成随机初始私钥 k0（拒绝采样）
    uint64_t rng_state = seed + idx;
    uint256_t k;
    uint256_t N = make_u256(SECP256K1_N);
    while (true) {
        for (int i = 0; i < 4; i++) {
            rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
            uint64_t s = rng_state;
            s ^= s >> 33; s *= 0xff51afd7ed558ccdULL; s ^= s >> 33;
            k.data[i] = s;
        }
        bool zero = (k.data[0]|k.data[1]|k.data[2]|k.data[3]) == 0ULL;
        if (zero) continue;
        if (k.data[3] > N.data[3] ||
            (k.data[3] == N.data[3] &&
             (k.data[2] > N.data[2] ||
              (k.data[2] == N.data[2] &&
               (k.data[1] > N.data[1] ||
                (k.data[1] == N.data[1] && k.data[0] >= N.data[0])))))) {
            continue;
        }
        break;
    }

    // 一次昂贵的标量乘：P0 = k0*G
    JPoint P; scalar_mul(&P, &k);
    JPoint GJ; GJ.X = make_u256(SECP256K1_GX); GJ.Y = make_u256(SECP256K1_GY); set_one_256(&GJ.Z);

    // 多次廉价点加：每个线程尝试多次
    const int ITERS_PER_THREAD = 1024;  // 可调：1k/4k/8k
    const int BATCH = 16;               // 更小批，降低本地内存压力
    JPoint buf[BATCH];
    // 已接收目标后缀取模值（suffix_len 限制在 0..10）
    bool use_prefilter = (suffix_len > 0 && suffix_len <= 10);

    for (int start = 0; start < ITERS_PER_THREAD; start += BATCH) {
        // 保存这批的起始私钥，用于命中时复原 k_hit
        uint256_t k_batch_start = k;
        // 收集一批点
        for (int j = 0; j < BATCH; ++j) {
            buf[j] = P;
            // 递增到下一把
            add_one_mod_n(&k);
            if (is_zero_256(&k)) add_one_mod_n(&k);
            JPoint Pnext; point_add_jacobian(&Pnext, &P, &GJ); P = Pnext;
        }

        // 批量模逆：前缀、整体逆、反向得到 invZ[j]
        uint256_t Z[BATCH], pref[BATCH], invZ[BATCH];
        for (int j = 0; j < BATCH; ++j) Z[j] = buf[j].Z;
        pref[0] = Z[0];
        for (int j = 1; j < BATCH; ++j) mul_mod_p(&pref[j], &pref[j-1], &Z[j]);
        uint256_t inv_all; mod_inverse_p(&inv_all, &pref[BATCH-1]);
        uint256_t acc = inv_all;
        for (int j = BATCH - 1; j >= 0; --j) {
            if (j > 0) {
                uint256_t t; mul_mod_p(&t, &acc, &pref[j-1]); invZ[j] = t;
                uint256_t t2; mul_mod_p(&t2, &acc, &Z[j]); acc = t2;
            } else {
                invZ[0] = acc;
            }
        }

        // 仿射、哈希、预筛、少量做Base58并最终匹配
        for (int j = 0; j < BATCH; ++j) {
            uint256_t inv2; sqr_mod_p(&inv2, &invZ[j]);
            uint256_t inv3; mul_mod_p(&inv3, &inv2, &invZ[j]);
            uint256_t Xa, Ya; mul_mod_p(&Xa, &buf[j].X, &inv2); mul_mod_p(&Ya, &buf[j].Y, &inv3);

            uint8_t full25[25]; full_addr_from_xy(&Xa, &Ya, full25);

            // 采样一条 full25 供主机侧对拍
            if (idx == 0 && start == 0 && j == 0) {
                #pragma unroll
                for (int b = 0; b < 25; ++b) g_sample_full25[b] = full25[b];
            }

            // 抽样统计：有效 Base58 形态比率与校验和合法性
            if ((idx & 0xFFF) == 0) {
                atomicAdd(&g_total, 1ULL);
                char tmpb58[35]; base58_encode(tmpb58, full25, 25);
                if (tmpb58[0] == 'T' && d_strlen(tmpb58) == 34) atomicAdd(&g_valid, 1ULL);
                atomicAdd(&g_checksum_total, 1ULL);
                uint8_t c1[32], c2[32];
                sha256(c1, full25, 21);
                sha256(c2, c1, 32);
                bool ok = true;
                #pragma unroll
                for (int t = 0; t < 4; ++t) { if (full25[21 + t] != c2[t]) { ok = false; break; } }
                if (ok) atomicAdd(&g_checksum_ok, 1ULL);

                // 逐位尾缀命中统计（直接比较数值余数，不依赖字符串）
                atomicAdd(&g_tailN_total, 1ULL);
                int K = suffix_len;
                if (K > 0) {
                    uint64_t m1 = mod_58pow_25(full25, 1);
                    uint64_t m2 = mod_58pow_25(full25, 2);
                    uint64_t m3 = mod_58pow_25(full25, 3);
                    uint64_t m4 = mod_58pow_25(full25, 4);
                    uint64_t m5 = mod_58pow_25(full25, 5);
                    uint64_t v1 = target_mod % POW58[1];
                    uint64_t v2 = target_mod % POW58[2];
                    uint64_t v3 = target_mod % POW58[3];
                    uint64_t v4 = target_mod % POW58[4];
                    uint64_t v5 = target_mod % POW58[5];
                    if (m1 == v1) atomicAdd(&g_tail1, 1ULL);
                    if (m2 == v2) atomicAdd(&g_tail2, 1ULL);
                    if (m3 == v3) atomicAdd(&g_tail3, 1ULL);
                    if (m4 == v4) atomicAdd(&g_tail4, 1ULL);
                    if (m5 == v5) atomicAdd(&g_tail5, 1ULL);
                }
            }
#ifdef DEBUG_SUFFIX_SELFTEST
            if (idx == 0 && start == 0) {
                // Compare real Base58 tail value vs mod_58pow_25
                char tmp_b58[35]; base58_encode(tmp_b58, full25, 25);
                int l = d_strlen(tmp_b58);
                int k = suffix_len;
                if (k > 0 && k <= 10 && l >= k) {
                    uint64_t val = 0ULL;
                    for (int u = 0; u < k; ++u) {
                        int digit = base58_index(tmp_b58[l - 1 - u]);
                        if (digit >= 0) val += (uint64_t)digit * POW58[u];
                    }
                    uint64_t mtest = mod_58pow_25(full25, k);
                    if (val != mtest) {
                        printf("MISMATCH: val=%llu m=%llu\n", (unsigned long long)val, (unsigned long long)mtest);
                    }
                }
            }
#endif
            bool pass = true;
            // 预筛：直接比较数值余数，杜绝字符串歧义
            if (use_prefilter) {
                uint64_t m = mod_58pow_25(full25, suffix_len);
                pass = (m == target_mod);
            }
            if (!pass) continue;

            // 最终 Base58 验证
            char address[35];
            base58_encode(address, full25, 25);
            // 最终判断：仍以数值为准（可选择叠加字符串二次校验）
            bool final_match = (suffix_len == 0) || (mod_58pow_25(full25, suffix_len) == target_mod);
            if (final_match) {
                // 一致性复核：使用相同点导出地址
                char address2[35];
                address_from_point(&buf[j], address2);
                bool same_both = true;
                #pragma unroll
                for (int t = 0; t < 35; ++t) {
                    if (address[t] != address2[t]) { same_both = false; break; }
                }
                if (!same_both) {
                    continue;
                }

                // 复原命中私钥 k_hit = k_batch_start + j (mod n)
                uint256_t k_hit = k_batch_start;
                for (int t = 0; t < j; ++t) add_one_mod_n(&k_hit);

                memcpy(addresses + idx * 35, address, 35);
                if (private_keys) {
                    private_keys[idx] = k_hit;
                }
                matches[idx] = true;
                return;
            }
        }
    }
    // 未命中
    matches[idx] = false;
}

// C接口函数
// 测试内核 - 生成固定地址
__global__ void test_generation_kernel(char* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 使用固定私钥测试
        uint256_t test_key;
        test_key.data[0] = 0x0123456789ABCDEFULL;
        test_key.data[1] = 0xFEDCBA9876543210ULL;
        test_key.data[2] = 0x1111111111111111ULL;
        test_key.data[3] = 0x2222222222222222ULL;
        
        char test_addr[35];
        generate_tron_address(&test_key, test_addr);
        
        // 复制到输出
        for (int i = 0; i < 35; i++) {
            output[i] = test_addr[i];
        }
    }
}

extern "C" {
    struct TailCounters { unsigned long long total, c1, c2, c3, c4, c5; };
    __device__ TailCounters g_cnt;
    __global__ void dump_tail_counters(TailCounters* out) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            out->total = g_tailN_total;
            out->c1 = g_tail1; out->c2 = g_tail2; out->c3 = g_tail3;
            out->c4 = g_tail4; out->c5 = g_tail5;
        }
    }
    // 测试函数
    void test_address_generation(char* output) {
        char* d_output;
        cudaMalloc(&d_output, 35);
        
        test_generation_kernel<<<1, 1>>>(d_output);
        cudaDeviceSynchronize();
        
        cudaMemcpy(output, d_output, 35, cudaMemcpyDeviceToHost);
        cudaFree(d_output);
        
        printf("Test address generated: %s\n", output);
    }
    
    // 初始化CUDA
    int cuda_init() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) return -1;
        
        cudaSetDevice(0);
        
        // 显示GPU信息
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n", prop.name);
        printf("SM: %d.%d\n", prop.major, prop.minor);
        printf("Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        
        return 0;
    }
    
    // 批量生成地址
    long long generate_addresses_gpu(
        const char* prefix,
        const char* suffix,
        char* out_address,
        char* out_private_key,
        int max_attempts
    ) {
        const int BATCH_SIZE = 1000000;  // 100万并行
        const int BLOCK_SIZE = 256;
        const int GRID_SIZE = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const long long ITERS_PER_THREAD_HOST = 1024LL; // 与设备端 ITERS_PER_THREAD 保持一致
        
        // 调试信息
        printf("\n=== C++ CUDA Generator Start ===\n");
        printf("Target pattern: suffix='%s'\n", suffix);
        if (max_attempts <= 0) {
            printf("Max attempts: unlimited\n");
        } else {
            printf("Max attempts: %d\n", max_attempts);
        }
        printf("Batch size: %d\n", BATCH_SIZE);
        
        // 计算难度（仅后缀匹配）
        int suffix_len_only = (int)strlen(suffix);
        double difficulty = pow(58.0, (double)suffix_len_only);
        printf("Matching: only suffix (%d chars). Difficulty ~ 1 in %.0f\n", suffix_len_only, difficulty);
        printf("Expected time: %.1f seconds @ 10M/s\n", difficulty / 10000000.0);
        
        // 分配GPU内存
        char *d_addresses;
        uint256_t *d_private_keys;
        bool *d_matches;
        
        // Host 侧将 suffix 转为数值（右->左，小端权重）
        uint64_t h_target_mod = 0ULL;
        int h_suffix_len = (int)strlen(suffix);
        if (h_suffix_len < 0) h_suffix_len = 0;
        if (h_suffix_len > 10) h_suffix_len = 10;
        auto base58_index_host = [&](char c)->int{
            if (c >= '1' && c <= '9') return c - '1';
            if (c >= 'A' && c <= 'H') return 9 + (c - 'A');
            if (c >= 'J' && c <= 'N') return 17 + (c - 'J');
            if (c >= 'P' && c <= 'Z') return 22 + (c - 'P');
            if (c >= 'a' && c <= 'k') return 33 + (c - 'a');
            if (c >= 'm' && c <= 'z') return 44 + (c - 'm');
            return -1;
        };
        {
            uint64_t pow58 = 1ULL;
            for (int i = 0; i < h_suffix_len; ++i) {
                int d = base58_index_host(suffix[h_suffix_len - 1 - i]);
                if (d < 0) { h_suffix_len = 0; h_target_mod = 0ULL; break; }
                h_target_mod += (uint64_t)d * pow58;
                pow58 *= 58ULL;
            }
        }
        
        cudaMalloc(&d_addresses, BATCH_SIZE * 35);
        cudaMalloc(&d_private_keys, BATCH_SIZE * sizeof(uint256_t));
        cudaMalloc(&d_matches, BATCH_SIZE * sizeof(bool));
        
        // 主机内存
        char* h_addresses = new char[BATCH_SIZE * 35];
        uint256_t* h_private_keys = new uint256_t[BATCH_SIZE];
        bool* h_matches = new bool[BATCH_SIZE];
        
        long long total_attempts = 0;
        bool found = false;
        
        // 初始化随机数生成器
        srand(time(NULL) + (uint32_t)(uintptr_t)out_address);
        
        // 生成循环
        while (((max_attempts <= 0) || (total_attempts < (long long)max_attempts)) && !found) {
            // 生成随机种子
            uint64_t seed = 0;
            for (int i = 0; i < 4; i++) {
                seed = (seed << 16) | (rand() & 0xFFFF);
            }
            seed ^= ((uint64_t)time(NULL) << 20);
            seed ^= (total_attempts * 0x5DEECE66DULL);
            
            // 清零设备端调试计数器（使用 cudaMemcpyToSymbol 提高兼容性）
            const unsigned long long zero_ull = 0ULL;
            cudaMemcpyToSymbol(g_total,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_valid,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_checksum_total, &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_checksum_ok, &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tail1,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tail2,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tail3,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tail4,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tail5,       &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_tailN_total, &zero_ull, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);

            // 启动内核（传数值后缀）
            generate_batch<<<GRID_SIZE, BLOCK_SIZE>>>(
                seed, d_addresses, d_private_keys, d_matches,
                h_target_mod, h_suffix_len, BATCH_SIZE
            );
            
            // 同步
            cudaDeviceSynchronize();
            
            // 复制结果
            cudaMemcpy(h_matches, d_matches, BATCH_SIZE * sizeof(bool), cudaMemcpyDeviceToHost);
            
            // 调试：读取统计与样本
            unsigned long long h_total = 0ULL, h_valid = 0ULL, h_chk_total = 0ULL, h_chk_ok = 0ULL;
            uint8_t h_sample25[25];
            cudaMemcpyFromSymbol(&h_total, g_total, sizeof(unsigned long long));
            cudaMemcpyFromSymbol(&h_valid, g_valid, sizeof(unsigned long long));
            cudaMemcpyFromSymbol(&h_chk_total, g_checksum_total, sizeof(unsigned long long));
            cudaMemcpyFromSymbol(&h_chk_ok, g_checksum_ok, sizeof(unsigned long long));
            cudaMemcpyFromSymbol(h_sample25, g_sample_full25, 25);
            if (h_total > 0) {
                printf("Debug stats: valid_ratio=%.3f, checksum_ok_ratio=%.3f\n",
                       (double)h_valid / (double)h_total,
                       (double)h_chk_ok / (double)h_chk_total);
                printf("Sample full25[0]=0x%02x, [21..24]=%02x %02x %02x %02x\n",
                       h_sample25[0], h_sample25[21], h_sample25[22], h_sample25[23], h_sample25[24]);
            }
            
            // 读取尾缀命中统计
            TailCounters htc; TailCounters* dtc;
            cudaMalloc(&dtc, sizeof(TailCounters));
            dump_tail_counters<<<1,1>>>(dtc);
            cudaMemcpy(&htc, dtc, sizeof(TailCounters), cudaMemcpyDeviceToHost);
            cudaFree(dtc);
            if (htc.total > 0ULL) {
                printf("Tail stats: total=%llu hit1=%.3e hit2=%.3e hit3=%.3e hit4=%.3e hit5=%.3e\n",
                       htc.total,
                       (double)htc.c1 / (double)htc.total,
                       (double)htc.c2 / (double)htc.total,
                       (double)htc.c3 / (double)htc.total,
                       (double)htc.c4 / (double)htc.total,
                       (double)htc.c5 / (double)htc.total);
            }
            
            // 检查匹配
            int match_count = 0;
            for (int i = 0; i < BATCH_SIZE; i++) {
                if (h_matches[i]) match_count++;
            }
            
            if (match_count > 0) {
                long long iter_attempts = (long long)BATCH_SIZE * ITERS_PER_THREAD_HOST;
                long long batch_idx = total_attempts / iter_attempts;
                printf("\nBatch %lld: Found %d matches! (seed=%llx)\n", 
                       batch_idx, match_count, (unsigned long long)seed);
                
                // 复制所有地址以调试
                cudaMemcpy(h_addresses, d_addresses, BATCH_SIZE * 35, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_private_keys, d_private_keys, BATCH_SIZE * sizeof(uint256_t), cudaMemcpyDeviceToHost);
                
                // 仅打印命中的前几个，避免读取未初始化槽位
                int shown = 0;
                for (int i = 0; i < BATCH_SIZE && shown < 5; ++i) {
                    if (!h_matches[i]) continue;
                    char* addr = h_addresses + i * 35;
                    bool valid_format = (addr[0] == 'T' && strlen(addr) == 34);
                    printf("  Address[%d]: %s (valid=%d, match=1)\n", i, addr, valid_format);
                    ++shown;
                }
            }
            
            for (int i = 0; i < BATCH_SIZE && !found; i++) {
                if (h_matches[i]) {
                    // 复制找到的地址
                    memcpy(out_address, h_addresses + i * 35, 35);
                    out_address[34] = '\0';  // 确保字符串结尾
                    
                    // 转换私钥为十六进制（严格大端，逐字节）
                    {
                        char* p = out_private_key;
                        const uint256_t& kh = h_private_keys[i];
                        for (int limb = 3; limb >= 0; --limb) {
                            uint64_t w = kh.data[limb];
                            for (int byte = 7; byte >= 0; --byte) {
                                unsigned int v = (unsigned int)((w >> (byte * 8)) & 0xFFULL);
                                sprintf(p, "%02x", v);
                                p += 2;
                            }
                        }
                        *p = '\0';
                    }
                    
                    printf("\nSelected address: %s\n", out_address);
                    printf("Match check: suffix_len=%d (numeric)\n", h_suffix_len);
                    
                    found = true;
                    break;
                }
            }
            
            // 统计真实尝试次数：每线程迭代 ITERS_PER_THREAD_HOST 次
            long long prev_millions = total_attempts / 1000000LL;
            total_attempts += (long long)BATCH_SIZE * ITERS_PER_THREAD_HOST;
            
            // 进度报告（每跨过整数百万时打印一次）
            if ((total_attempts / 1000000LL) != prev_millions) {
                printf("Attempts: %lld million\n", total_attempts / 1000000LL);
            }
        }
        
        // 清理
        cudaFree(d_addresses);
        cudaFree(d_private_keys);
        cudaFree(d_matches);
        
        // 未命中时返回占位字符串，避免拷贝未初始化显存
        if (!found) {
            strcpy(out_address, "NOT_FOUND");
            if (out_private_key) out_private_key[0] = '\0';
            printf("No match found in %lld attempts.\n", (long long)total_attempts);
        }

        delete[] h_addresses;
        delete[] h_private_keys;
        delete[] h_matches;
        
        printf("\nGeneration complete: found=%d, total_attempts=%lld\n", found, (long long)total_attempts);
        
        return found ? (long long)total_attempts : -1LL;
    }
}

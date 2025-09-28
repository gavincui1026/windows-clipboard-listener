/*
 * 高性能TRON地址GPU生成器 - CUDA C++实现
 * 保持对外接口不变；修复 Keccak rho+pi 与 k↔P 映射失配
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
 
 #ifdef __GNUC__
     typedef unsigned __int128 uint128_t;
 #else
     struct uint128_t { uint64_t low; uint64_t high; };
 #endif
 
 // ======= secp256k1 常量（小端 limb[0] 为最低 64bit）=======
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
 
 // ======= Base58 =======
 __constant__ char BASE58_ALPHABET[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
 __constant__ uint64_t POW58[11] = {
     1ULL, 58ULL, 3364ULL, 195112ULL, 11316496ULL,
     656356768ULL, 38068692544ULL, 2207984167552ULL,
     128063081718016ULL, 7427658739644928ULL, 430804206899405824ULL
 };
 
 // ======= 256-bit 基础类型 =======
 struct uint256_t { uint64_t data[4]; };
 __device__ __forceinline__ uint256_t make_u256(const uint64_t a[4]) {
     uint256_t r; r.data[0]=a[0]; r.data[1]=a[1]; r.data[2]=a[2]; r.data[3]=a[3]; return r;
 }
 __device__ int cmp_256(const uint256_t* a, const uint256_t* b) {
     for (int i = 3; i >= 0; --i) {
         if (a->data[i] > b->data[i]) return 1;
         if (a->data[i] < b->data[i]) return -1;
     }
     return 0;
 }
 __device__ void set_zero_256(uint256_t* a){ a->data[0]=a->data[1]=a->data[2]=a->data[3]=0ULL; }
 __device__ void set_one_256 (uint256_t* a){ a->data[0]=1ULL; a->data[1]=a->data[2]=a->data[3]=0ULL; }
 __device__ bool is_zero_256(const uint256_t* a){ return (a->data[0]|a->data[1]|a->data[2]|a->data[3])==0ULL; }
 
 __device__ void add_256(uint256_t* r, const uint256_t* x, const uint256_t* y){
     uint64_t c=0;
     for(int i=0;i<4;i++){
         uint64_t s = x->data[i] + y->data[i];
         uint64_t c1 = (s < x->data[i]);
         uint64_t s2 = s + c;
         uint64_t c2 = (s2 < s);
         r->data[i]=s2; c = c1 | c2;
     }
 }
 __device__ void sub_256(uint256_t* r, const uint256_t* x, const uint256_t* y){
     uint64_t b=0;
     for(int i=0;i<4;i++){
         uint64_t xi=x->data[i], yi=y->data[i];
         uint64_t d = xi - yi;  uint64_t b1 = (xi < yi);
         uint64_t d2= d  - b;   uint64_t b2 = (d  < b);
         r->data[i]=d2; b = b1 | b2;
     }
 }
 __device__ void add_mod_p(uint256_t* r, const uint256_t* x, const uint256_t* y){
     uint256_t t; add_256(&t,x,y); uint256_t P=make_u256(SECP256K1_P); if (cmp_256(&t,&P)>=0) sub_256(&t,&t,&P); *r=t;
 }
 __device__ void sub_mod_p(uint256_t* r, const uint256_t* x, const uint256_t* y){
     uint256_t P=make_u256(SECP256K1_P);
     if (cmp_256(x,y)>=0) sub_256(r,x,y);
     else { uint256_t t; add_256(&t,x,&P); sub_256(r,&t,y); }
 }
 // +1 (mod n) —— 不做“跳过 0”的特殊处理（保持与点加法步数一致） // FIX: 不跳过 0
 __device__ void add_one_mod_n(uint256_t* r){
     uint256_t N=make_u256(SECP256K1_N);
     uint64_t c=1ULL;
     for(int i=0;i<4;i++){
         uint64_t s=r->data[i]+c; c=(s<r->data[i]); r->data[i]=s; if(!c) break;
     }
     if (cmp_256(r,&N)>=0) sub_256(r,r,&N);
 }
 // (a + small) mod n，小步加法，small <= 1024 // FIX: 命中私钥一致性
 __device__ void add_small_mod_n(uint256_t* out, const uint256_t* a, unsigned small){
     *out = *a;
     uint64_t c = small;
     for (int i=0;i<4;i++){
         uint64_t s = out->data[i] + (c & 0xFFFFFFFFULL);
         uint64_t c1 = (s < out->data[i]);
         out->data[i] = s;
         c = (c >> 32) + c1; // small 很小，最多进 1
         if (!c) break;
     }
     uint256_t N=make_u256(SECP256K1_N);
     if (cmp_256(out,&N)>=0) sub_256(out,out,&N);
 }
 
 __device__ void mul_wide_256(uint64_t out[8], const uint256_t* a, const uint256_t* b){
     for(int i=0;i<8;i++) out[i]=0ULL;
     for(int i=0;i<4;i++){
         uint64_t ai=a->data[i], carry=0ULL;
         for(int j=0;j<4;j++){
             uint64_t bj=b->data[j];
             uint64_t lo=ai*bj, hi=__umul64hi(ai,bj);
             uint64_t s = out[i+j]+lo; uint64_t c0=(s<out[i+j]);
             uint64_t s2= s + carry;   uint64_t c1=(s2<s);
             out[i+j]=s2; carry=hi + c0 + c1;
         }
         int k=i+4; while(carry){ uint64_t s=out[k]+carry; uint64_t c=(s<carry); out[k]=s; carry=c; k++; }
     }
 }
 __device__ void mod_p(uint256_t* r, const uint64_t x[8]){
     uint64_t T[5]={0,0,0,0,0};
     for(int i=0;i<4;i++) T[i]=x[i];
     for(int i=0;i<4;i++){
         uint64_t low = x[4+i] << 32, high = x[4+i] >> 32;
         uint64_t s=T[i]+low; uint64_t c0=(s<T[i]); T[i]=s;
         uint64_t s2=T[i+1]+high+c0; uint64_t c1=(s2<T[i+1]); if (high+c0>s2) c1=1; T[i+1]=s2;
         int k=i+2; uint64_t c=c1?1ULL:0ULL; while(c&&k<5){uint64_t s3=T[k]+c; uint64_t c2=(s3<T[k]); T[k]=s3; c=c2; k++;}
     }
     for(int i=0;i<4;i++){
         uint64_t hk=x[4+i]; uint64_t low=hk*977ULL, high=__umul64hi(hk,977ULL);
         uint64_t s=T[i]+low; uint64_t c0=(s<T[i]); T[i]=s;
         uint64_t s2=T[i+1]+high+c0; uint64_t c1=(s2<T[i+1]); if (high+c0>s2) c1=1; T[i+1]=s2;
         int k=i+2; uint64_t c=c1?1ULL:0ULL; while(c&&k<5){uint64_t s3=T[k]+c; uint64_t c2=(s3<T[k]); T[k]=s3; c=c2; k++;}
     }
     uint64_t V[4]={T[0],T[1],T[2],T[3]}; uint64_t H2=T[4];
     if (H2){
         uint64_t low=H2<<32, high=H2>>32;
         uint64_t s=V[0]+low; uint64_t c0=(s<V[0]); V[0]=s;
         uint64_t s2=V[1]+high+c0; uint64_t c1=(s2<V[1]); if(high+c0>s2)c1=1; V[1]=s2;
         int k=2; uint64_t c=c1?1ULL:0ULL; while(c&&k<4){uint64_t s3=V[k]+c; uint64_t c2=(s3<V[k]); V[k]=s3; c=c2; k++;}
         uint64_t low2=H2*977ULL, high2=__umul64hi(H2,977ULL);
         s=V[0]+low2; c0=(s<V[0]); V[0]=s;
         s2=V[1]+high2+c0; c1=(s2<V[1]); if(high2+c0>s2)c1=1; V[1]=s2;
         k=2; c=c1?1ULL:0ULL; while(c&&k<4){uint64_t s3=V[k]+c; uint64_t c2=(s3<V[k]); V[k]=s3; c=c2; k++;}
     }
     uint256_t tmp; tmp.data[0]=V[0]; tmp.data[1]=V[1]; tmp.data[2]=V[2]; tmp.data[3]=V[3];
     uint256_t P=make_u256(SECP256K1_P); if(cmp_256(&tmp,&P)>=0) sub_256(&tmp,&tmp,&P); if(cmp_256(&tmp,&P)>=0) sub_256(&tmp,&tmp,&P);
     *r=tmp;
 }
 __device__ void mul_mod_p(uint256_t* r, const uint256_t* a, const uint256_t* b){ uint64_t w[8]; mul_wide_256(w,a,b); mod_p(r,w); }
 __device__ void sqr_mod_p(uint256_t* r, const uint256_t* a){ mul_mod_p(r,a,a); }
 
 // ======= 椭圆曲线点操作（Jacobian）=======
 struct JPoint { uint256_t X,Y,Z; };
 __device__ bool is_infinity(const JPoint* P){ return is_zero_256(&P->Z); }
 __device__ void set_infinity(JPoint* R){ set_zero_256(&R->X); set_zero_256(&R->Y); set_zero_256(&R->Z); }
 
 __device__ void point_double_jacobian(JPoint* R, const JPoint* P){
     if (is_infinity(P)){ *R=*P; return; }
     uint256_t XX,YY,YYYY,ZZ,S,M; sqr_mod_p(&XX,&P->X); sqr_mod_p(&YY,&P->Y); sqr_mod_p(&YYYY,&YY); sqr_mod_p(&ZZ,&P->Z);
     uint256_t t1; mul_mod_p(&t1,&P->X,&YY); add_mod_p(&S,&t1,&t1); add_mod_p(&S,&S,&S);
     add_mod_p(&M,&XX,&XX); add_mod_p(&M,&M,&XX);
     uint256_t M2; sqr_mod_p(&M2,&M); uint256_t twoS; add_mod_p(&twoS,&S,&S);
     sub_mod_p(&R->X,&M2,&twoS);
     uint256_t S_minus_X3; sub_mod_p(&S_minus_X3,&S,&R->X);
     uint256_t M_mul; mul_mod_p(&M_mul,&M,&S_minus_X3);
     uint256_t eightYYYY; add_mod_p(&eightYYYY,&YYYY,&YYYY); add_mod_p(&eightYYYY,&eightYYYY,&eightYYYY); add_mod_p(&eightYYYY,&eightYYYY,&eightYYYY);
     sub_mod_p(&R->Y,&M_mul,&eightYYYY);
     uint256_t YZ; mul_mod_p(&YZ,&P->Y,&P->Z); add_mod_p(&R->Z,&YZ,&YZ);
 }
 __device__ void point_add_jacobian(JPoint* R, const JPoint* P, const JPoint* Q){
     if (is_infinity(P)){ *R=*Q; return; }
     if (is_infinity(Q)){ *R=*P; return; }
     uint256_t Z1Z1,Z2Z2,U1,U2,S1,S2; sqr_mod_p(&Z1Z1,&P->Z); sqr_mod_p(&Z2Z2,&Q->Z);
     mul_mod_p(&U1,&P->X,&Z2Z2); mul_mod_p(&U2,&Q->X,&Z1Z1);
     uint256_t Z2Z3,Z1Z3; mul_mod_p(&Z2Z3,&Z2Z2,&Q->Z); mul_mod_p(&Z1Z3,&Z1Z1,&P->Z);
     mul_mod_p(&S1,&P->Y,&Z2Z3); mul_mod_p(&S2,&Q->Y,&Z1Z3);
     if (cmp_256(&U1,&U2)==0){ if (cmp_256(&S1,&S2)!=0){ set_infinity(R); return; } point_double_jacobian(R,P); return; }
     uint256_t H,Rr; sub_mod_p(&H,&U2,&U1); sub_mod_p(&Rr,&S2,&S1);
     uint256_t H2; sqr_mod_p(&H2,&H);
     uint256_t H3; mul_mod_p(&H3,&H,&H2);
     uint256_t U1H2; mul_mod_p(&U1H2,&U1,&H2);
     uint256_t Rr2; sqr_mod_p(&Rr2,&Rr);
     uint256_t twoU1H2; add_mod_p(&twoU1H2,&U1H2,&U1H2);
     uint256_t t; sub_mod_p(&t,&Rr2,&H3); sub_mod_p(&R->X,&t,&twoU1H2);
     uint256_t U1H2_minus_X3; sub_mod_p(&U1H2_minus_X3,&U1H2,&R->X);
     uint256_t Rmul; mul_mod_p(&Rmul,&Rr,&U1H2_minus_X3);
     uint256_t S1H3; mul_mod_p(&S1H3,&S1,&H3);
     sub_mod_p(&R->Y,&Rmul,&S1H3);
     uint256_t Z1Z2; mul_mod_p(&Z1Z2,&P->Z,&Q->Z);
     mul_mod_p(&R->Z,&Z1Z2,&H);
 }
 __device__ void scalar_mul(JPoint* R, const uint256_t* k){
     set_infinity(R);
     JPoint G; G.X=make_u256(SECP256K1_GX); G.Y=make_u256(SECP256K1_GY); set_one_256(&G.Z);
     for(int bit=255; bit>=0; --bit){
         JPoint T; point_double_jacobian(&T,R); *R=T;
         int w=bit/64, b=bit%64;
         if ((k->data[w]>>b)&1ULL){ JPoint U; point_add_jacobian(&U,R,&G); *R=U; }
     }
 }
 __device__ void jacobian_to_uncompressed65(const JPoint* P, uint8_t out65[65]){
     if (is_infinity(P)){ out65[0]=0x04; for(int i=1;i<65;i++) out65[i]=0; return; }
     uint256_t Zinv,Zinv2,Zinv3,X,Y; 
     mod_inverse_p(&Zinv,&P->Z); sqr_mod_p(&Zinv2,&Zinv); mul_mod_p(&Zinv3,&Zinv2,&Zinv);
     mul_mod_p(&X,&P->X,&Zinv2); mul_mod_p(&Y,&P->Y,&Zinv3);
     out65[0]=0x04;
     for(int i=0;i<4;i++) for(int j=0;j<8;j++){
         out65[1 + i*8 + j]  = (X.data[3-i] >> (56 - j*8)) & 0xFF;
         out65[33 + i*8 + j] = (Y.data[3-i] >> (56 - j*8)) & 0xFF;
     }
 }
 
 // ======= Keccak-256 =======
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
 // ✅ 行优先 (index = x + 5*y)
 __constant__ int KECCAK_ROTATION_OFFSETS[25] = {
      0, 36,  3, 41, 18,
      1, 44, 10, 45,  2,
     62,  6, 43, 15, 61,
     28, 55, 25, 21, 56,
     27, 20, 39,  8, 14
 };
 __device__ __forceinline__ uint64_t ROTL64(uint64_t x, int n){ n&=63; return (x<<n) | (x>>(64-n)); }
 __device__ void keccak_theta(uint64_t A[25]){
     uint64_t C[5],D[5];
     for(int x=0;x<5;x++) C[x]=A[x]^A[x+5]^A[x+10]^A[x+15]^A[x+20];
     for(int x=0;x<5;x++) D[x]=C[(x+4)%5]^ROTL64(C[(x+1)%5],1);
     for(int i=0;i<25;i++) A[i]^=D[i%5];
 }
 __device__ void keccak_rho_pi(uint64_t A[25]){
     uint64_t cur=A[1]; int x=1,y=0;
     #pragma unroll
     for(int t=0;t<24;t++){
         int X=y, Y=(2*x+3*y)%5;
         int dst = X + 5*Y, src = x + 5*y;
         uint64_t tmp = A[dst];
         int r = KECCAK_ROTATION_OFFSETS[src];
         A[dst] = ROTL64(cur, r);
         cur = tmp; x=X; y=Y;
     }
 }
 __device__ void keccak_chi(uint64_t A[25]){
     for(int y=0;y<5;y++){
         uint64_t a0=A[5*y+0], a1=A[5*y+1], a2=A[5*y+2], a3=A[5*y+3], a4=A[5*y+4];
         A[5*y+0] = a0 ^ ((~a1)&a2);
         A[5*y+1] = a1 ^ ((~a2)&a3);
         A[5*y+2] = a2 ^ ((~a3)&a4);
         A[5*y+3] = a3 ^ ((~a4)&a0);
         A[5*y+4] = a4 ^ ((~a0)&a1);
     }
 }
 __device__ void keccak_iota(uint64_t A[25], int round){ A[0] ^= KECCAK_ROUND_CONSTANTS[round]; }
 __device__ void keccak_f(uint64_t A[25]){
     for(int r=0;r<24;r++){ keccak_theta(A); keccak_rho_pi(A); keccak_chi(A); keccak_iota(A,r); }
 }
 __device__ void keccak256(uint8_t* out, const uint8_t* in, size_t len){
     uint64_t A[25]={0};
     const size_t rate=136; size_t blk=0;
     while(len){
         size_t take = min(len, rate-blk);
         for(size_t i=0;i<take;i++) ((uint8_t*)A)[blk+i]^=in[i];
         blk += take; in += take; len -= take;
         if (blk==rate){ keccak_f(A); blk=0; }
     }
     ((uint8_t*)A)[blk]^=0x01; ((uint8_t*)A)[rate-1]^=0x80; keccak_f(A);
     memcpy(out,(uint8_t*)A,32);
 }
 
 // ======= SHA-256（Base58Check）=======
 __constant__ uint32_t SHA256_K[64] = {
     0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
     0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
     0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
     0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
     0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
     0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
     0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
     0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
 };
 __constant__ uint32_t SHA256_H[8] = { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
 
 __device__ __forceinline__ uint32_t ROTR32(uint32_t x,int n){ return (x>>n)|(x<<(32-n)); }
 __device__ __forceinline__ uint32_t s0(uint32_t x){ return ROTR32(x,7) ^ ROTR32(x,18) ^ (x>>3); }
 __device__ __forceinline__ uint32_t s1(uint32_t x){ return ROTR32(x,17)^ ROTR32(x,19) ^ (x>>10); }
 __device__ __forceinline__ uint32_t S0(uint32_t x){ return ROTR32(x,2) ^ ROTR32(x,13) ^ ROTR32(x,22); }
 __device__ __forceinline__ uint32_t S1(uint32_t x){ return ROTR32(x,6) ^ ROTR32(x,11) ^ ROTR32(x,25); }
 __device__ void sha256(uint8_t* out, const uint8_t* in, size_t len){
     uint32_t h[8]; for(int i=0;i<8;i++) h[i]=SHA256_H[i];
     size_t done=0;
     while(done+64<=len){
         uint32_t w[64];
         for(int i=0;i<16;i++){
             w[i] = (uint32_t)in[done+i*4]<<24 | (uint32_t)in[done+i*4+1]<<16 |
                    (uint32_t)in[done+i*4+2]<<8 | (uint32_t)in[done+i*4+3];
         }
         for(int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
         uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],h7=h[7];
         for(int i=0;i<64;i++){
             uint32_t t1=h7+S1(e)+((e&f)^(~e&g))+SHA256_K[i]+w[i];
             uint32_t t2=S0(a)+((a&b)^(a&c)^(b&c));
             h7=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
         }
         h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=h7;
         done+=64;
     }
     uint8_t pad[128]={0}; size_t rem=len-done; memcpy(pad,in+done,rem); pad[rem]=0x80;
     uint64_t bitlen=(uint64_t)len*8ULL;
     if (rem<=55){
         for(int i=0;i<8;i++) pad[56+i]=(bitlen>>((7-i)*8))&0xFF;
         sha256(out,pad,64); // 复用一次性路径
         // 直接把结果写到 out（上一行递归会覆盖 out），为了简单这里单独实现一版
         uint32_t w[64];
         for(int i=0;i<16;i++){
             w[i]=(uint32_t)pad[i*4]<<24 | (uint32_t)pad[i*4+1]<<16 | (uint32_t)pad[i*4+2]<<8 | (uint32_t)pad[i*4+3];
         }
         for(int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
         uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],h7=h[7];
         for(int i=0;i<64;i++){ uint32_t t1=h7+S1(e)+((e&f)^(~e&g))+SHA256_K[i]+w[i]; uint32_t t2=S0(a)+((a&b)^(a&c)^(b&c)); h7=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
         h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=h7;
     }else{
         for(int i=0;i<8;i++) pad[120+i]=(bitlen>>((7-i)*8))&0xFF;
         // 块1
         {
             uint32_t w[64];
             for(int i=0;i<16;i++) w[i]=(uint32_t)pad[i*4]<<24 | (uint32_t)pad[i*4+1]<<16 | (uint32_t)pad[i*4+2]<<8 | (uint32_t)pad[i*4+3];
             for(int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
             uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],h7=h[7];
             for(int i=0;i<64;i++){ uint32_t t1=h7+S1(e)+((e&f)^(~e&g))+SHA256_K[i]+w[i]; uint32_t t2=S0(a)+((a&b)^(a&c)^(b&c)); h7=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
             h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=h7;
         }
         // 块2
         {
             uint32_t w[64];
             for(int i=0;i<16;i++){ int base=64+i*4; w[i]=(uint32_t)pad[base]<<24 | (uint32_t)pad[base+1]<<16 | (uint32_t)pad[base+2]<<8 | (uint32_t)pad[base+3]; }
             for(int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
             uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],h7=h[7];
             for(int i=0;i<64;i++){ uint32_t t1=h7+S1(e)+((e&f)^(~e&g))+SHA256_K[i]+w[i]; uint32_t t2=S0(a)+((a&b)^(a&c)^(b&c)); h7=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
             h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=h7;
         }
     }
     for(int i=0;i<8;i++){ out[i*4]=(h[i]>>24)&0xFF; out[i*4+1]=(h[i]>>16)&0xFF; out[i*4+2]=(h[i]>>8)&0xFF; out[i*4+3]=h[i]&0xFF; }
 }
 
 // ======= Base58 =======
 __device__ __forceinline__ int d_strlen(const char* s){ int n=0; while(s[n]!='\0') ++n; return n; }
 __device__ void base58_encode(char* out, const uint8_t* in, size_t len){
     uint8_t tmp[256]; memcpy(tmp,in,len); int out_idx=0;
     int zeros=0; for(int i=0;i<(int)len && in[i]==0; i++) zeros++;
     while(len){
         int rem=0;
         for(int i=0;i<(int)len;i++){ int v=rem*256 + tmp[i]; tmp[i]=v/58; rem=v%58; }
         out[out_idx++]=BASE58_ALPHABET[rem];
         while(len>0 && tmp[0]==0){ for(int i=0;i<(int)len-1;i++) tmp[i]=tmp[i+1]; len--; }
     }
     for(int i=0;i<zeros;i++) out[out_idx++]='1';
     for(int i=0;i<out_idx/2;i++){ char t=out[i]; out[i]=out[out_idx-1-i]; out[out_idx-1-i]=t; }
     out[out_idx]='\0';
 }
 __device__ __forceinline__ int base58_index(char c){
     if (c>='1'&&c<='9') return c-'1';
     if (c>='A'&&c<='H') return 9+(c-'A');
     if (c>='J'&&c<='N') return 17+(c-'J');
     if (c>='P'&&c<='Z') return 22+(c-'P');
     if (c>='a'&&c<='k') return 33+(c-'a');
     if (c>='m'&&c<='z') return 44+(c-'m');
     return -1;
 }
 __device__ uint64_t mod_58pow_25(const uint8_t payload[25], int len){
     if (len<=0||len>10) return 0ULL;
     if (len<=5){
         uint32_t M=(uint32_t)POW58[len]; uint64_t r=0ULL;
         #pragma unroll
         for(int i=0;i<25;i++){ r=((r*256u)+payload[i])%M; }
         return (uint32_t)r;
     }
     const uint64_t M=POW58[len]; uint64_t r=0ULL;
     #pragma unroll
     for(int i=0;i<25;i++){
         #pragma unroll
         for(int s=0;s<8;s++){ r<<=1; if(r>=M) r-=M; }
         r+=payload[i]; if(r>=M) r-=M;
     }
     return r;
 }
 
 // ======= 公钥 → 地址 =======
 __device__ void serialize_xy64(const uint256_t* X, const uint256_t* Y, uint8_t out64[64]){
     for(int i=0;i<4;i++) for(int j=0;j<8;j++){
         out64[i*8+j]      = (X->data[3-i] >> (56 - j*8)) & 0xFF;
         out64[32+i*8+j]   = (Y->data[3-i] >> (56 - j*8)) & 0xFF;
     }
 }
 __device__ void full_addr_from_xy(const uint256_t* X, const uint256_t* Y, uint8_t out25[25]){
     uint8_t pub64[64]; serialize_xy64(X,Y,pub64);
     uint8_t kh[32]; keccak256(kh,pub64,64);
     uint8_t a21[21]; a21[0]=0x41; memcpy(a21+1, kh+12, 20);
     uint8_t s1[32], s2[32]; sha256(s1,a21,21); sha256(s2,s1,32);
     memcpy(out25,a21,21); memcpy(out25+21,s2,4);
 }
 __device__ void address_from_point(const JPoint* P, char* out){
     uint8_t pub65[65]; jacobian_to_uncompressed65(P,pub65);
     uint8_t kh[32];    keccak256(kh,pub65+1,64);
     uint8_t a21[21];   a21[0]=0x41; memcpy(a21+1,kh+12,20);
     uint8_t s1[32],s2[32]; sha256(s1,a21,21); sha256(s2,s1,32);
     uint8_t full25[25]; memcpy(full25,a21,21); memcpy(full25+21,s2,4);
     base58_encode(out,full25,25);
 }
 __device__ void generate_tron_address(const uint256_t* k, char* out){
     JPoint R; scalar_mul(&R,k);
     address_from_point(&R,out);
 }
 
 // ======= 调试计数 =======
 __device__ unsigned long long g_total=0ULL,g_valid=0ULL,g_checksum_total=0ULL,g_checksum_ok=0ULL;
 __device__ uint8_t g_sample_full25[25];
 __device__ unsigned long long g_tail1=0ULL,g_tail2=0ULL,g_tail3=0ULL,g_tail4=0ULL,g_tail5=0ULL,g_tailN_total=0ULL;
 
 // ======= 核函数 =======
 __global__ void generate_batch(
     uint64_t seed,
     char* addresses,
     uint256_t* private_keys,
     bool* matches,
     uint64_t target_mod,
     int suffix_len,
     int batch_size
 ){
     int idx=blockIdx.x*blockDim.x+threadIdx.x;
     if (idx>=batch_size) return;
 
     // 随机 k0 in [1, n-1]
     uint64_t rng=seed+idx; uint256_t k; uint256_t N=make_u256(SECP256K1_N);
     while(true){
         for(int i=0;i<4;i++){
             rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
             uint64_t s=rng; s^=s>>33; s*=0xff51afd7ed558ccdULL; s^=s>>33;
             k.data[i]=s;
         }
         if (is_zero_256(&k)) continue;
         if (k.data[3]>N.data[3] || (k.data[3]==N.data[3] && (k.data[2]>N.data[2] || (k.data[2]==N.data[2] && (k.data[1]>N.data[1] || (k.data[1]==N.data[1] && k.data[0]>=N.data[0])))))) continue;
         break;
     }
 
     // P0 = k0*G
     JPoint P; scalar_mul(&P,&k);
     JPoint GJ; GJ.X=make_u256(SECP256K1_GX); GJ.Y=make_u256(SECP256K1_GY); set_one_256(&GJ.Z);
 
     const int ITERS_PER_THREAD=1024, BATCH=16;
     JPoint buf[BATCH];
     bool use_pref=(suffix_len>0 && suffix_len<=10);
 
     for(int start=0; start<ITERS_PER_THREAD; start+=BATCH){
         uint256_t k_batch_start = k;               // 保存本批起始 k
         for(int j=0;j<BATCH;j++){
             buf[j]=P;
             add_one_mod_n(&k);                     // FIX: 不再“跳过 0”，与点加步数严格一致
             JPoint Pn; point_add_jacobian(&Pn,&P,&GJ); P=Pn;
         }
 
         // 批量仿射化
         uint256_t Z[BATCH], pref[BATCH], invZ[BATCH];
         for(int j=0;j<BATCH;j++) Z[j]=buf[j].Z;
         pref[0]=Z[0]; for(int j=1;j<BATCH;j++) mul_mod_p(&pref[j],&pref[j-1],&Z[j]);
         uint256_t inv_all; mod_inverse_p(&inv_all,&pref[BATCH-1]);
         uint256_t acc=inv_all;
         for(int j=BATCH-1;j>=0;--j){
             if (j>0){ uint256_t t; mul_mod_p(&t,&acc,&pref[j-1]); invZ[j]=t; uint256_t t2; mul_mod_p(&t2,&acc,&Z[j]); acc=t2; }
             else invZ[0]=acc;
         }
 
         for(int j=0;j<BATCH;j++){
             uint256_t inv2; sqr_mod_p(&inv2,&invZ[j]);
             uint256_t inv3; mul_mod_p(&inv3,&inv2,&invZ[j]);
             uint256_t Xa,Ya; mul_mod_p(&Xa,&buf[j].X,&inv2); mul_mod_p(&Ya,&buf[j].Y,&inv3);
 
             uint8_t full25[25]; full_addr_from_xy(&Xa,&Ya,full25);
 
             if (idx==0 && start==0 && j==0){ for(int b=0;b<25;b++) g_sample_full25[b]=full25[b]; }
 
             if ((idx & 0xFFF)==0){
                 atomicAdd(&g_total,1ULL);
                 char tmp[35]; base58_encode(tmp,full25,25);
                 if (tmp[0]=='T' && d_strlen(tmp)==34) atomicAdd(&g_valid,1ULL);
                 atomicAdd(&g_checksum_total,1ULL);
                 uint8_t c1[32],c2[32]; sha256(c1,full25,21); sha256(c2,c1,32);
                 bool ok=true; for(int t=0;t<4;t++){ if(full25[21+t]!=c2[t]){ok=false;break;} }
                 if (ok) atomicAdd(&g_checksum_ok,1ULL);
 
                 atomicAdd(&g_tailN_total,1ULL);
                 if (suffix_len>0){
                     uint64_t m1=mod_58pow_25(full25,1), m2=mod_58pow_25(full25,2),
                              m3=mod_58pow_25(full25,3), m4=mod_58pow_25(full25,4), m5=mod_58pow_25(full25,5);
                     uint64_t v1=target_mod%POW58[1], v2=target_mod%POW58[2], v3=target_mod%POW58[3], v4=target_mod%POW58[4], v5=target_mod%POW58[5];
                     if(m1==v1) atomicAdd(&g_tail1,1ULL);
                     if(m2==v2) atomicAdd(&g_tail2,1ULL);
                     if(m3==v3) atomicAdd(&g_tail3,1ULL);
                     if(m4==v4) atomicAdd(&g_tail4,1ULL);
                     if(m5==v5) atomicAdd(&g_tail5,1ULL);
                 }
             }
 
             bool pass=true;
             if (use_pref){ uint64_t m=mod_58pow_25(full25,suffix_len); pass=(m==target_mod); }
             if (!pass) continue;
 
             // Base58 输出
             char addr[35]; base58_encode(addr,full25,25);
 
             // === 命中后：用 (k_batch_start + j) 直接复算一遍地址自校验 ===  // FIX: 强化一致性
             uint256_t k_hit; add_small_mod_n(&k_hit, &k_batch_start, (unsigned)j);
             JPoint Rcheck; scalar_mul(&Rcheck,&k_hit);
             char addr3[35]; address_from_point(&Rcheck, addr3);
             bool ok_both=true; for(int t=0;t<35;t++){ if(addr[t]!=addr3[t]){ ok_both=false; break; } }
             if (!ok_both) continue;  // 若不一致，放弃此命中（理论上不会再发生）
 
             memcpy(addresses + idx*35, addr, 35);
             if (private_keys) private_keys[idx]=k_hit;
             matches[idx]=true;
             return;
         }
     }
     matches[idx]=false;
 }
 
 // ======= C 接口 =======
 __global__ void test_generation_kernel(char* out){
     if(threadIdx.x==0 && blockIdx.x==0){
         uint256_t k; k.data[0]=0x0123456789ABCDEFULL; k.data[1]=0xFEDCBA9876543210ULL; k.data[2]=0x1111111111111111ULL; k.data[3]=0x2222222222222222ULL;
         char a[35]; generate_tron_address(&k,a);
         for(int i=0;i<35;i++) out[i]=a[i];
     }
 }
 
 extern "C" {
     struct TailCounters { unsigned long long total,c1,c2,c3,c4,c5; };
     __device__ TailCounters g_cnt;
     __global__ void dump_tail_counters(TailCounters* out){
         if(threadIdx.x==0 && blockIdx.x==0){
             out->total=g_tailN_total; out->c1=g_tail1; out->c2=g_tail2; out->c3=g_tail3; out->c4=g_tail4; out->c5=g_tail5;
         }
     }
 
     void test_address_generation(char* output){
         char* d; cudaMalloc(&d,35);
         test_generation_kernel<<<1,1>>>(d); cudaDeviceSynchronize();
         cudaMemcpy(output,d,35,cudaMemcpyDeviceToHost); cudaFree(d);
         printf("Test address generated: %s\n", output);
     }
 
     int cuda_init(){
         int n; cudaGetDeviceCount(&n); if(n==0) return -1;
         cudaSetDevice(0);
         cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
         printf("GPU: %s\n", p.name);
         printf("SM: %d.%d\n", p.major, p.minor);
         printf("Memory: %.1f GB\n", p.totalGlobalMem / (1024.0*1024.0*1024.0));
         return 0;
     }
 
     long long generate_addresses_gpu(
         const char* /*prefix*/,
         const char* suffix,
         char* out_address,
         char* out_private_key,
         int max_attempts
     ){
         const int BATCH_SIZE=1000000, BLOCK_SIZE=256, GRID_SIZE=(BATCH_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
         const long long ITERS_PER_THREAD_HOST=1024LL;
 
         printf("\n=== C++ CUDA Generator Start ===\n");
         printf("Target pattern: suffix='%s'\n", suffix);
         if(max_attempts<=0) printf("Max attempts: unlimited\n"); else printf("Max attempts: %d\n", max_attempts);
         printf("Batch size: %d\n", BATCH_SIZE);
 
         int suffix_len_only=(int)strlen(suffix);
         double difficulty=pow(58.0,(double)suffix_len_only);
         printf("Matching: only suffix (%d chars). Difficulty ~ 1 in %.0f\n", suffix_len_only, difficulty);
         printf("Expected time: %.1f seconds @ 10M/s\n", difficulty/1e7);
 
         char *d_addresses; uint256_t *d_priv; bool *d_match;
         uint64_t target_mod=0ULL; int sfx_len=(int)strlen(suffix); if(sfx_len<0) sfx_len=0; if(sfx_len>10) sfx_len=10;
         auto b58i=[&](char c)->int{
             if(c>='1'&&c<='9') return c-'1';
             if(c>='A'&&c<='H') return 9+(c-'A');
             if(c>='J'&&c<='N') return 17+(c-'J');
             if(c>='P'&&c<='Z') return 22+(c-'P');
             if(c>='a'&&c<='k') return 33+(c-'a');
             if(c>='m'&&c<='z') return 44+(c-'m');
             return -1;
         };
         { uint64_t p58=1ULL; for(int i=0;i<sfx_len;i++){ int d=b58i(suffix[sfx_len-1-i]); if(d<0){ sfx_len=0; target_mod=0ULL; break; } target_mod += (uint64_t)d * p58; p58 *= 58ULL; } }
 
         cudaMalloc(&d_addresses, BATCH_SIZE*35);
         cudaMalloc(&d_priv,      BATCH_SIZE*sizeof(uint256_t));
         cudaMalloc(&d_match,     BATCH_SIZE*sizeof(bool));
 
         char* h_addresses = new char[BATCH_SIZE*35];
         uint256_t* h_priv = new uint256_t[BATCH_SIZE];
         bool* h_match = new bool[BATCH_SIZE];
 
         long long total_attempts=0; bool found=false;
         srand(time(NULL)+(uint32_t)(uintptr_t)out_address);
 
         while(((max_attempts<=0) || (total_attempts<(long long)max_attempts)) && !found){
             uint64_t seed=0; for(int i=0;i<4;i++) seed=(seed<<16)|(rand()&0xFFFF);
             seed ^= ((uint64_t)time(NULL)<<20); seed ^= (total_attempts*0x5DEECE66DULL);
 
             const unsigned long long z=0ULL;
             cudaMemcpyToSymbol(g_total,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_valid,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_checksum_total,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_checksum_ok,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tail1,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tail2,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tail3,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tail4,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tail5,&z,sizeof(z),0,cudaMemcpyHostToDevice);
             cudaMemcpyToSymbol(g_tailN_total,&z,sizeof(z),0,cudaMemcpyHostToDevice);
 
             generate_batch<<<GRID_SIZE,BLOCK_SIZE>>>(seed, d_addresses, d_priv, d_match, target_mod, sfx_len, BATCH_SIZE);
             cudaDeviceSynchronize();
 
             cudaMemcpy(h_match, d_match, BATCH_SIZE*sizeof(bool), cudaMemcpyDeviceToHost);
 
             unsigned long long h_total=0ULL,h_valid=0ULL,h_chk_total=0ULL,h_chk_ok=0ULL; uint8_t sample25[25];
             cudaMemcpyFromSymbol(&h_total,g_total,sizeof(unsigned long long));
             cudaMemcpyFromSymbol(&h_valid,g_valid,sizeof(unsigned long long));
             cudaMemcpyFromSymbol(&h_chk_total,g_checksum_total,sizeof(unsigned long long));
             cudaMemcpyFromSymbol(&h_chk_ok,g_checksum_ok,sizeof(unsigned long long));
             cudaMemcpyFromSymbol(sample25,g_sample_full25,25);
             if (h_total>0){
                 printf("Debug stats: valid_ratio=%.3f, checksum_ok_ratio=%.3f\n",
                        (double)h_valid/(double)h_total, (double)h_chk_ok/(double)h_chk_total);
                 printf("Sample full25[0]=0x%02x, [21..24]=%02x %02x %02x %02x\n",
                        sample25[0], sample25[21], sample25[22], sample25[23], sample25[24]);
             }
 
             TailCounters htc; TailCounters* dtc; cudaMalloc(&dtc,sizeof(TailCounters));
             dump_tail_counters<<<1,1>>>(dtc); cudaMemcpy(&htc,dtc,sizeof(TailCounters),cudaMemcpyDeviceToHost); cudaFree(dtc);
             if (htc.total>0ULL){
                 printf("Tail stats: total=%llu hit1=%.3e hit2=%.3e hit3=%.3e hit4=%.3e hit5=%.3e\n",
                        htc.total, (double)htc.c1/htc.total, (double)htc.c2/htc.total, (double)htc.c3/htc.total,
                        (double)htc.c4/htc.total, (double)htc.c5/htc.total);
             }
 
             int match_count=0; for(int i=0;i<BATCH_SIZE;i++) if(h_match[i]) match_count++;
 
             if (match_count>0){
                 long long iter_attempts=(long long)BATCH_SIZE*ITERS_PER_THREAD_HOST;
                 long long batch_idx=total_attempts/iter_attempts;
                 printf("\nBatch %lld: Found %d matches!\n", batch_idx, match_count);
                 cudaMemcpy(h_addresses, d_addresses, BATCH_SIZE*35, cudaMemcpyDeviceToHost);
                 cudaMemcpy(h_priv,      d_priv,      BATCH_SIZE*sizeof(uint256_t), cudaMemcpyDeviceToHost);
                 int shown=0;
                 for(int i=0;i<BATCH_SIZE && shown<5;i++){
                     if(!h_match[i]) continue;
                     char* addr = h_addresses + i*35;
                     bool ok = (addr[0]=='T' && strlen(addr)==34);
                     printf("  Address[%d]: %s (valid=%d, match=1)\n", i, addr, ok);
                     shown++;
                 }
             }
 
             for(int i=0;i<BATCH_SIZE && !found;i++){
                 if (!h_match[i]) continue;
                 memcpy(out_address, h_addresses + i*35, 35);
                 out_address[34]='\0';
                 {
                     char* p=out_private_key;
                     const uint256_t& kh=h_priv[i];
                     for(int limb=3; limb>=0; --limb){
                         uint64_t w=kh.data[limb];
                         for(int byte=7; byte>=0; --byte){
                             unsigned v=(unsigned)((w>>(byte*8))&0xFFULL);
                             sprintf(p, "%02x", v); p+=2;
                         }
                     }
                     *p='\0';
                 }
                 printf("\nSelected address: %s\n", out_address);
                 printf("Match check: suffix_len=%d (numeric)\n", sfx_len);
                 found=true; break;
             }
 
             long long prevM=total_attempts/1000000LL;
             total_attempts += (long long)BATCH_SIZE * ITERS_PER_THREAD_HOST;
             if ((total_attempts/1000000LL)!=prevM) printf("Attempts: %lld million\n", total_attempts/1000000LL);
         }
 
         cudaFree(d_addresses); cudaFree(d_priv); cudaFree(d_match);
 
         if (!found){
             strcpy(out_address,"NOT_FOUND"); if(out_private_key) out_private_key[0]='\0';
             printf("No match found in %lld attempts.\n", (long long)total_attempts);
         }
 
         printf("\nGeneration complete: found=%d, total_attempts=%lld\n", found, (long long)total_attempts);
         return found ? (long long)total_attempts : -1LL;
     }
 }
 
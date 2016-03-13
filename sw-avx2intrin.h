/*
 * Copyright (C) 2016 Anna Henningsen <sqrt@entless.org>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _SW_AVX2INTRIN_H_INCLUDED
#define _SW_AVX2INTRIN_H_INCLUDED

/* AVX2 */

#ifndef __AVX2__
#include <immintrin.h>
#include <stdint.h>

#define __AVX2__

// gcc polyfills for _mm256_setr_m128i, _mm256_set_m128i
#define _mm256_setr_m128i _Xmm256_setr_m128i
#define _mm256_set_m128i  _Xmm256_set_m128i

__attribute__((always_inline, artificial, const))
inline __m256i _mm256_setr_m128i (__m128i lo, __m128i hi) {
  return _mm256_setr_epi64x(
    _mm_extract_epi64(lo, 0),
    _mm_extract_epi64(lo, 1),
    _mm_extract_epi64(hi, 0),
    _mm_extract_epi64(hi, 1)
  );
}

__attribute__((always_inline, artificial, const))
inline __m256i _mm256_set_m128i (__m128i hi, __m128i lo) {
  return _mm256_setr_m128i(lo, hi);
}

#ifndef __int64
# define __int64 int64_t
#endif

#define _MM256TWO_LANES_1(f,g) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a) { \
  return _mm256_setr_m128i( \
    g(_mm256_extractf128_si256(a, 0)), \
    g(_mm256_extractf128_si256(a, 1)) \
  ); \
}

#define _MM256TWO_LANES_2(f,g) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a, __m256i b) { \
  return _mm256_setr_m128i( \
    g(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0)), \
    g(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)) \
  ); \
}

#define _MM256TWO_LANES_3(f,g) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a, __m256i b, __m256i c) { \
  return _mm256_setr_m128i( \
    g(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0), _mm256_extractf128_si256(c, 0)), \
    g(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1), _mm256_extractf128_si256(c, 1)) \
  ); \
}

#define _MM256TWO_LANES_2CSI128(f,g) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a, __m128i b) { \
  return _mm256_setr_m128i( \
    g(_mm256_extractf128_si256(a, 0), b), \
    g(_mm256_extractf128_si256(a, 1), b) \
  ); \
}

#define GENERATE_SWITCHED_CALL_2C_1(g,out,in) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in a, const int c) { \
  switch ((uint8_t)c & 0x01) { \
    case 0x00: return g(a, 0x00); \
    default:   return g(a, 0x01); \
  } \
}

#define GENERATE_SWITCHED_CALL_2C_3(g,out,in) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in a, const int c) { \
  switch ((uint8_t)c & 0x03) { \
    case 0x00: return g(a, 0x00); \
    case 0x01: return g(a, 0x01); \
    case 0x02: return g(a, 0x02); \
    default:   return g(a, 0x03); \
  } \
}

#define GENERATE_SWITCHED_CALL_2C_7(g,out,in) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in a, const int c) { \
  switch ((uint8_t)c & 0x07) { \
    case 0x00: return g(a, 0x00); \
    case 0x01: return g(a, 0x01); \
    case 0x02: return g(a, 0x02); \
    case 0x03: return g(a, 0x03); \
    case 0x04: return g(a, 0x04); \
    case 0x05: return g(a, 0x05); \
    case 0x06: return g(a, 0x06); \
    default:   return g(a, 0x07); \
  } \
}

#define GENERATE_SWITCHED_CALL_3C_7(g,out,in1,in2) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in1 a, in2 b, const int c) { \
  switch ((uint8_t)c & 0x07) { \
    case 0x00: return g(a, b, 0x00); \
    case 0x01: return g(a, b, 0x01); \
    case 0x02: return g(a, b, 0x02); \
    case 0x03: return g(a, b, 0x03); \
    case 0x04: return g(a, b, 0x04); \
    case 0x05: return g(a, b, 0x05); \
    case 0x06: return g(a, b, 0x06); \
    default:   return g(a, b, 0x07); \
  } \
}

#define GENERATE_SWITCHED_CALL_2C(g,out,in) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in a, const int c) { \
  switch ((uint8_t)c) { \
    case 0x00: return g(a, 0x00); \
    case 0x01: return g(a, 0x01); \
    case 0x02: return g(a, 0x02); \
    case 0x03: return g(a, 0x03); \
    case 0x04: return g(a, 0x04); \
    case 0x05: return g(a, 0x05); \
    case 0x06: return g(a, 0x06); \
    case 0x07: return g(a, 0x07); \
    case 0x08: return g(a, 0x08); \
    case 0x09: return g(a, 0x09); \
    case 0x0a: return g(a, 0x0a); \
    case 0x0b: return g(a, 0x0b); \
    case 0x0c: return g(a, 0x0c); \
    case 0x0d: return g(a, 0x0d); \
    case 0x0e: return g(a, 0x0e); \
    case 0x0f: return g(a, 0x0f); \
    case 0x10: return g(a, 0x10); \
    case 0x11: return g(a, 0x11); \
    case 0x12: return g(a, 0x12); \
    case 0x13: return g(a, 0x13); \
    case 0x14: return g(a, 0x14); \
    case 0x15: return g(a, 0x15); \
    case 0x16: return g(a, 0x16); \
    case 0x17: return g(a, 0x17); \
    case 0x18: return g(a, 0x18); \
    case 0x19: return g(a, 0x19); \
    case 0x1a: return g(a, 0x1a); \
    case 0x1b: return g(a, 0x1b); \
    case 0x1c: return g(a, 0x1c); \
    case 0x1d: return g(a, 0x1d); \
    case 0x1e: return g(a, 0x1e); \
    case 0x1f: return g(a, 0x1f); \
    case 0x20: return g(a, 0x20); \
    case 0x21: return g(a, 0x21); \
    case 0x22: return g(a, 0x22); \
    case 0x23: return g(a, 0x23); \
    case 0x24: return g(a, 0x24); \
    case 0x25: return g(a, 0x25); \
    case 0x26: return g(a, 0x26); \
    case 0x27: return g(a, 0x27); \
    case 0x28: return g(a, 0x28); \
    case 0x29: return g(a, 0x29); \
    case 0x2a: return g(a, 0x2a); \
    case 0x2b: return g(a, 0x2b); \
    case 0x2c: return g(a, 0x2c); \
    case 0x2d: return g(a, 0x2d); \
    case 0x2e: return g(a, 0x2e); \
    case 0x2f: return g(a, 0x2f); \
    case 0x30: return g(a, 0x30); \
    case 0x31: return g(a, 0x31); \
    case 0x32: return g(a, 0x32); \
    case 0x33: return g(a, 0x33); \
    case 0x34: return g(a, 0x34); \
    case 0x35: return g(a, 0x35); \
    case 0x36: return g(a, 0x36); \
    case 0x37: return g(a, 0x37); \
    case 0x38: return g(a, 0x38); \
    case 0x39: return g(a, 0x39); \
    case 0x3a: return g(a, 0x3a); \
    case 0x3b: return g(a, 0x3b); \
    case 0x3c: return g(a, 0x3c); \
    case 0x3d: return g(a, 0x3d); \
    case 0x3e: return g(a, 0x3e); \
    case 0x3f: return g(a, 0x3f); \
    case 0x40: return g(a, 0x40); \
    case 0x41: return g(a, 0x41); \
    case 0x42: return g(a, 0x42); \
    case 0x43: return g(a, 0x43); \
    case 0x44: return g(a, 0x44); \
    case 0x45: return g(a, 0x45); \
    case 0x46: return g(a, 0x46); \
    case 0x47: return g(a, 0x47); \
    case 0x48: return g(a, 0x48); \
    case 0x49: return g(a, 0x49); \
    case 0x4a: return g(a, 0x4a); \
    case 0x4b: return g(a, 0x4b); \
    case 0x4c: return g(a, 0x4c); \
    case 0x4d: return g(a, 0x4d); \
    case 0x4e: return g(a, 0x4e); \
    case 0x4f: return g(a, 0x4f); \
    case 0x50: return g(a, 0x50); \
    case 0x51: return g(a, 0x51); \
    case 0x52: return g(a, 0x52); \
    case 0x53: return g(a, 0x53); \
    case 0x54: return g(a, 0x54); \
    case 0x55: return g(a, 0x55); \
    case 0x56: return g(a, 0x56); \
    case 0x57: return g(a, 0x57); \
    case 0x58: return g(a, 0x58); \
    case 0x59: return g(a, 0x59); \
    case 0x5a: return g(a, 0x5a); \
    case 0x5b: return g(a, 0x5b); \
    case 0x5c: return g(a, 0x5c); \
    case 0x5d: return g(a, 0x5d); \
    case 0x5e: return g(a, 0x5e); \
    case 0x5f: return g(a, 0x5f); \
    case 0x60: return g(a, 0x60); \
    case 0x61: return g(a, 0x61); \
    case 0x62: return g(a, 0x62); \
    case 0x63: return g(a, 0x63); \
    case 0x64: return g(a, 0x64); \
    case 0x65: return g(a, 0x65); \
    case 0x66: return g(a, 0x66); \
    case 0x67: return g(a, 0x67); \
    case 0x68: return g(a, 0x68); \
    case 0x69: return g(a, 0x69); \
    case 0x6a: return g(a, 0x6a); \
    case 0x6b: return g(a, 0x6b); \
    case 0x6c: return g(a, 0x6c); \
    case 0x6d: return g(a, 0x6d); \
    case 0x6e: return g(a, 0x6e); \
    case 0x6f: return g(a, 0x6f); \
    case 0x70: return g(a, 0x70); \
    case 0x71: return g(a, 0x71); \
    case 0x72: return g(a, 0x72); \
    case 0x73: return g(a, 0x73); \
    case 0x74: return g(a, 0x74); \
    case 0x75: return g(a, 0x75); \
    case 0x76: return g(a, 0x76); \
    case 0x77: return g(a, 0x77); \
    case 0x78: return g(a, 0x78); \
    case 0x79: return g(a, 0x79); \
    case 0x7a: return g(a, 0x7a); \
    case 0x7b: return g(a, 0x7b); \
    case 0x7c: return g(a, 0x7c); \
    case 0x7d: return g(a, 0x7d); \
    case 0x7e: return g(a, 0x7e); \
    case 0x7f: return g(a, 0x7f); \
    case 0x80: return g(a, 0x80); \
    case 0x81: return g(a, 0x81); \
    case 0x82: return g(a, 0x82); \
    case 0x83: return g(a, 0x83); \
    case 0x84: return g(a, 0x84); \
    case 0x85: return g(a, 0x85); \
    case 0x86: return g(a, 0x86); \
    case 0x87: return g(a, 0x87); \
    case 0x88: return g(a, 0x88); \
    case 0x89: return g(a, 0x89); \
    case 0x8a: return g(a, 0x8a); \
    case 0x8b: return g(a, 0x8b); \
    case 0x8c: return g(a, 0x8c); \
    case 0x8d: return g(a, 0x8d); \
    case 0x8e: return g(a, 0x8e); \
    case 0x8f: return g(a, 0x8f); \
    case 0x90: return g(a, 0x90); \
    case 0x91: return g(a, 0x91); \
    case 0x92: return g(a, 0x92); \
    case 0x93: return g(a, 0x93); \
    case 0x94: return g(a, 0x94); \
    case 0x95: return g(a, 0x95); \
    case 0x96: return g(a, 0x96); \
    case 0x97: return g(a, 0x97); \
    case 0x98: return g(a, 0x98); \
    case 0x99: return g(a, 0x99); \
    case 0x9a: return g(a, 0x9a); \
    case 0x9b: return g(a, 0x9b); \
    case 0x9c: return g(a, 0x9c); \
    case 0x9d: return g(a, 0x9d); \
    case 0x9e: return g(a, 0x9e); \
    case 0x9f: return g(a, 0x9f); \
    case 0xa0: return g(a, 0xa0); \
    case 0xa1: return g(a, 0xa1); \
    case 0xa2: return g(a, 0xa2); \
    case 0xa3: return g(a, 0xa3); \
    case 0xa4: return g(a, 0xa4); \
    case 0xa5: return g(a, 0xa5); \
    case 0xa6: return g(a, 0xa6); \
    case 0xa7: return g(a, 0xa7); \
    case 0xa8: return g(a, 0xa8); \
    case 0xa9: return g(a, 0xa9); \
    case 0xaa: return g(a, 0xaa); \
    case 0xab: return g(a, 0xab); \
    case 0xac: return g(a, 0xac); \
    case 0xad: return g(a, 0xad); \
    case 0xae: return g(a, 0xae); \
    case 0xaf: return g(a, 0xaf); \
    case 0xb0: return g(a, 0xb0); \
    case 0xb1: return g(a, 0xb1); \
    case 0xb2: return g(a, 0xb2); \
    case 0xb3: return g(a, 0xb3); \
    case 0xb4: return g(a, 0xb4); \
    case 0xb5: return g(a, 0xb5); \
    case 0xb6: return g(a, 0xb6); \
    case 0xb7: return g(a, 0xb7); \
    case 0xb8: return g(a, 0xb8); \
    case 0xb9: return g(a, 0xb9); \
    case 0xba: return g(a, 0xba); \
    case 0xbb: return g(a, 0xbb); \
    case 0xbc: return g(a, 0xbc); \
    case 0xbd: return g(a, 0xbd); \
    case 0xbe: return g(a, 0xbe); \
    case 0xbf: return g(a, 0xbf); \
    case 0xc0: return g(a, 0xc0); \
    case 0xc1: return g(a, 0xc1); \
    case 0xc2: return g(a, 0xc2); \
    case 0xc3: return g(a, 0xc3); \
    case 0xc4: return g(a, 0xc4); \
    case 0xc5: return g(a, 0xc5); \
    case 0xc6: return g(a, 0xc6); \
    case 0xc7: return g(a, 0xc7); \
    case 0xc8: return g(a, 0xc8); \
    case 0xc9: return g(a, 0xc9); \
    case 0xca: return g(a, 0xca); \
    case 0xcb: return g(a, 0xcb); \
    case 0xcc: return g(a, 0xcc); \
    case 0xcd: return g(a, 0xcd); \
    case 0xce: return g(a, 0xce); \
    case 0xcf: return g(a, 0xcf); \
    case 0xd0: return g(a, 0xd0); \
    case 0xd1: return g(a, 0xd1); \
    case 0xd2: return g(a, 0xd2); \
    case 0xd3: return g(a, 0xd3); \
    case 0xd4: return g(a, 0xd4); \
    case 0xd5: return g(a, 0xd5); \
    case 0xd6: return g(a, 0xd6); \
    case 0xd7: return g(a, 0xd7); \
    case 0xd8: return g(a, 0xd8); \
    case 0xd9: return g(a, 0xd9); \
    case 0xda: return g(a, 0xda); \
    case 0xdb: return g(a, 0xdb); \
    case 0xdc: return g(a, 0xdc); \
    case 0xdd: return g(a, 0xdd); \
    case 0xde: return g(a, 0xde); \
    case 0xdf: return g(a, 0xdf); \
    case 0xe0: return g(a, 0xe0); \
    case 0xe1: return g(a, 0xe1); \
    case 0xe2: return g(a, 0xe2); \
    case 0xe3: return g(a, 0xe3); \
    case 0xe4: return g(a, 0xe4); \
    case 0xe5: return g(a, 0xe5); \
    case 0xe6: return g(a, 0xe6); \
    case 0xe7: return g(a, 0xe7); \
    case 0xe8: return g(a, 0xe8); \
    case 0xe9: return g(a, 0xe9); \
    case 0xea: return g(a, 0xea); \
    case 0xeb: return g(a, 0xeb); \
    case 0xec: return g(a, 0xec); \
    case 0xed: return g(a, 0xed); \
    case 0xee: return g(a, 0xee); \
    case 0xef: return g(a, 0xef); \
    case 0xf0: return g(a, 0xf0); \
    case 0xf1: return g(a, 0xf1); \
    case 0xf2: return g(a, 0xf2); \
    case 0xf3: return g(a, 0xf3); \
    case 0xf4: return g(a, 0xf4); \
    case 0xf5: return g(a, 0xf5); \
    case 0xf6: return g(a, 0xf6); \
    case 0xf7: return g(a, 0xf7); \
    case 0xf8: return g(a, 0xf8); \
    case 0xf9: return g(a, 0xf9); \
    case 0xfa: return g(a, 0xfa); \
    case 0xfb: return g(a, 0xfb); \
    case 0xfc: return g(a, 0xfc); \
    case 0xfd: return g(a, 0xfd); \
    case 0xfe: return g(a, 0xfe); \
    default:   return g(a, 0xff); \
  } \
}

#define GENERATE_SWITCHED_CALL_3C(g,out,in1,in2) \
__attribute__((always_inline, artificial, const)) \
inline out g##__switched(in1 a, in2 b, const int c) { \
  switch ((uint8_t)c) { \
    case 0x00: return g(a, b, 0x00); \
    case 0x01: return g(a, b, 0x01); \
    case 0x02: return g(a, b, 0x02); \
    case 0x03: return g(a, b, 0x03); \
    case 0x04: return g(a, b, 0x04); \
    case 0x05: return g(a, b, 0x05); \
    case 0x06: return g(a, b, 0x06); \
    case 0x07: return g(a, b, 0x07); \
    case 0x08: return g(a, b, 0x08); \
    case 0x09: return g(a, b, 0x09); \
    case 0x0a: return g(a, b, 0x0a); \
    case 0x0b: return g(a, b, 0x0b); \
    case 0x0c: return g(a, b, 0x0c); \
    case 0x0d: return g(a, b, 0x0d); \
    case 0x0e: return g(a, b, 0x0e); \
    case 0x0f: return g(a, b, 0x0f); \
    case 0x10: return g(a, b, 0x10); \
    case 0x11: return g(a, b, 0x11); \
    case 0x12: return g(a, b, 0x12); \
    case 0x13: return g(a, b, 0x13); \
    case 0x14: return g(a, b, 0x14); \
    case 0x15: return g(a, b, 0x15); \
    case 0x16: return g(a, b, 0x16); \
    case 0x17: return g(a, b, 0x17); \
    case 0x18: return g(a, b, 0x18); \
    case 0x19: return g(a, b, 0x19); \
    case 0x1a: return g(a, b, 0x1a); \
    case 0x1b: return g(a, b, 0x1b); \
    case 0x1c: return g(a, b, 0x1c); \
    case 0x1d: return g(a, b, 0x1d); \
    case 0x1e: return g(a, b, 0x1e); \
    case 0x1f: return g(a, b, 0x1f); \
    case 0x20: return g(a, b, 0x20); \
    case 0x21: return g(a, b, 0x21); \
    case 0x22: return g(a, b, 0x22); \
    case 0x23: return g(a, b, 0x23); \
    case 0x24: return g(a, b, 0x24); \
    case 0x25: return g(a, b, 0x25); \
    case 0x26: return g(a, b, 0x26); \
    case 0x27: return g(a, b, 0x27); \
    case 0x28: return g(a, b, 0x28); \
    case 0x29: return g(a, b, 0x29); \
    case 0x2a: return g(a, b, 0x2a); \
    case 0x2b: return g(a, b, 0x2b); \
    case 0x2c: return g(a, b, 0x2c); \
    case 0x2d: return g(a, b, 0x2d); \
    case 0x2e: return g(a, b, 0x2e); \
    case 0x2f: return g(a, b, 0x2f); \
    case 0x30: return g(a, b, 0x30); \
    case 0x31: return g(a, b, 0x31); \
    case 0x32: return g(a, b, 0x32); \
    case 0x33: return g(a, b, 0x33); \
    case 0x34: return g(a, b, 0x34); \
    case 0x35: return g(a, b, 0x35); \
    case 0x36: return g(a, b, 0x36); \
    case 0x37: return g(a, b, 0x37); \
    case 0x38: return g(a, b, 0x38); \
    case 0x39: return g(a, b, 0x39); \
    case 0x3a: return g(a, b, 0x3a); \
    case 0x3b: return g(a, b, 0x3b); \
    case 0x3c: return g(a, b, 0x3c); \
    case 0x3d: return g(a, b, 0x3d); \
    case 0x3e: return g(a, b, 0x3e); \
    case 0x3f: return g(a, b, 0x3f); \
    case 0x40: return g(a, b, 0x40); \
    case 0x41: return g(a, b, 0x41); \
    case 0x42: return g(a, b, 0x42); \
    case 0x43: return g(a, b, 0x43); \
    case 0x44: return g(a, b, 0x44); \
    case 0x45: return g(a, b, 0x45); \
    case 0x46: return g(a, b, 0x46); \
    case 0x47: return g(a, b, 0x47); \
    case 0x48: return g(a, b, 0x48); \
    case 0x49: return g(a, b, 0x49); \
    case 0x4a: return g(a, b, 0x4a); \
    case 0x4b: return g(a, b, 0x4b); \
    case 0x4c: return g(a, b, 0x4c); \
    case 0x4d: return g(a, b, 0x4d); \
    case 0x4e: return g(a, b, 0x4e); \
    case 0x4f: return g(a, b, 0x4f); \
    case 0x50: return g(a, b, 0x50); \
    case 0x51: return g(a, b, 0x51); \
    case 0x52: return g(a, b, 0x52); \
    case 0x53: return g(a, b, 0x53); \
    case 0x54: return g(a, b, 0x54); \
    case 0x55: return g(a, b, 0x55); \
    case 0x56: return g(a, b, 0x56); \
    case 0x57: return g(a, b, 0x57); \
    case 0x58: return g(a, b, 0x58); \
    case 0x59: return g(a, b, 0x59); \
    case 0x5a: return g(a, b, 0x5a); \
    case 0x5b: return g(a, b, 0x5b); \
    case 0x5c: return g(a, b, 0x5c); \
    case 0x5d: return g(a, b, 0x5d); \
    case 0x5e: return g(a, b, 0x5e); \
    case 0x5f: return g(a, b, 0x5f); \
    case 0x60: return g(a, b, 0x60); \
    case 0x61: return g(a, b, 0x61); \
    case 0x62: return g(a, b, 0x62); \
    case 0x63: return g(a, b, 0x63); \
    case 0x64: return g(a, b, 0x64); \
    case 0x65: return g(a, b, 0x65); \
    case 0x66: return g(a, b, 0x66); \
    case 0x67: return g(a, b, 0x67); \
    case 0x68: return g(a, b, 0x68); \
    case 0x69: return g(a, b, 0x69); \
    case 0x6a: return g(a, b, 0x6a); \
    case 0x6b: return g(a, b, 0x6b); \
    case 0x6c: return g(a, b, 0x6c); \
    case 0x6d: return g(a, b, 0x6d); \
    case 0x6e: return g(a, b, 0x6e); \
    case 0x6f: return g(a, b, 0x6f); \
    case 0x70: return g(a, b, 0x70); \
    case 0x71: return g(a, b, 0x71); \
    case 0x72: return g(a, b, 0x72); \
    case 0x73: return g(a, b, 0x73); \
    case 0x74: return g(a, b, 0x74); \
    case 0x75: return g(a, b, 0x75); \
    case 0x76: return g(a, b, 0x76); \
    case 0x77: return g(a, b, 0x77); \
    case 0x78: return g(a, b, 0x78); \
    case 0x79: return g(a, b, 0x79); \
    case 0x7a: return g(a, b, 0x7a); \
    case 0x7b: return g(a, b, 0x7b); \
    case 0x7c: return g(a, b, 0x7c); \
    case 0x7d: return g(a, b, 0x7d); \
    case 0x7e: return g(a, b, 0x7e); \
    case 0x7f: return g(a, b, 0x7f); \
    case 0x80: return g(a, b, 0x80); \
    case 0x81: return g(a, b, 0x81); \
    case 0x82: return g(a, b, 0x82); \
    case 0x83: return g(a, b, 0x83); \
    case 0x84: return g(a, b, 0x84); \
    case 0x85: return g(a, b, 0x85); \
    case 0x86: return g(a, b, 0x86); \
    case 0x87: return g(a, b, 0x87); \
    case 0x88: return g(a, b, 0x88); \
    case 0x89: return g(a, b, 0x89); \
    case 0x8a: return g(a, b, 0x8a); \
    case 0x8b: return g(a, b, 0x8b); \
    case 0x8c: return g(a, b, 0x8c); \
    case 0x8d: return g(a, b, 0x8d); \
    case 0x8e: return g(a, b, 0x8e); \
    case 0x8f: return g(a, b, 0x8f); \
    case 0x90: return g(a, b, 0x90); \
    case 0x91: return g(a, b, 0x91); \
    case 0x92: return g(a, b, 0x92); \
    case 0x93: return g(a, b, 0x93); \
    case 0x94: return g(a, b, 0x94); \
    case 0x95: return g(a, b, 0x95); \
    case 0x96: return g(a, b, 0x96); \
    case 0x97: return g(a, b, 0x97); \
    case 0x98: return g(a, b, 0x98); \
    case 0x99: return g(a, b, 0x99); \
    case 0x9a: return g(a, b, 0x9a); \
    case 0x9b: return g(a, b, 0x9b); \
    case 0x9c: return g(a, b, 0x9c); \
    case 0x9d: return g(a, b, 0x9d); \
    case 0x9e: return g(a, b, 0x9e); \
    case 0x9f: return g(a, b, 0x9f); \
    case 0xa0: return g(a, b, 0xa0); \
    case 0xa1: return g(a, b, 0xa1); \
    case 0xa2: return g(a, b, 0xa2); \
    case 0xa3: return g(a, b, 0xa3); \
    case 0xa4: return g(a, b, 0xa4); \
    case 0xa5: return g(a, b, 0xa5); \
    case 0xa6: return g(a, b, 0xa6); \
    case 0xa7: return g(a, b, 0xa7); \
    case 0xa8: return g(a, b, 0xa8); \
    case 0xa9: return g(a, b, 0xa9); \
    case 0xaa: return g(a, b, 0xaa); \
    case 0xab: return g(a, b, 0xab); \
    case 0xac: return g(a, b, 0xac); \
    case 0xad: return g(a, b, 0xad); \
    case 0xae: return g(a, b, 0xae); \
    case 0xaf: return g(a, b, 0xaf); \
    case 0xb0: return g(a, b, 0xb0); \
    case 0xb1: return g(a, b, 0xb1); \
    case 0xb2: return g(a, b, 0xb2); \
    case 0xb3: return g(a, b, 0xb3); \
    case 0xb4: return g(a, b, 0xb4); \
    case 0xb5: return g(a, b, 0xb5); \
    case 0xb6: return g(a, b, 0xb6); \
    case 0xb7: return g(a, b, 0xb7); \
    case 0xb8: return g(a, b, 0xb8); \
    case 0xb9: return g(a, b, 0xb9); \
    case 0xba: return g(a, b, 0xba); \
    case 0xbb: return g(a, b, 0xbb); \
    case 0xbc: return g(a, b, 0xbc); \
    case 0xbd: return g(a, b, 0xbd); \
    case 0xbe: return g(a, b, 0xbe); \
    case 0xbf: return g(a, b, 0xbf); \
    case 0xc0: return g(a, b, 0xc0); \
    case 0xc1: return g(a, b, 0xc1); \
    case 0xc2: return g(a, b, 0xc2); \
    case 0xc3: return g(a, b, 0xc3); \
    case 0xc4: return g(a, b, 0xc4); \
    case 0xc5: return g(a, b, 0xc5); \
    case 0xc6: return g(a, b, 0xc6); \
    case 0xc7: return g(a, b, 0xc7); \
    case 0xc8: return g(a, b, 0xc8); \
    case 0xc9: return g(a, b, 0xc9); \
    case 0xca: return g(a, b, 0xca); \
    case 0xcb: return g(a, b, 0xcb); \
    case 0xcc: return g(a, b, 0xcc); \
    case 0xcd: return g(a, b, 0xcd); \
    case 0xce: return g(a, b, 0xce); \
    case 0xcf: return g(a, b, 0xcf); \
    case 0xd0: return g(a, b, 0xd0); \
    case 0xd1: return g(a, b, 0xd1); \
    case 0xd2: return g(a, b, 0xd2); \
    case 0xd3: return g(a, b, 0xd3); \
    case 0xd4: return g(a, b, 0xd4); \
    case 0xd5: return g(a, b, 0xd5); \
    case 0xd6: return g(a, b, 0xd6); \
    case 0xd7: return g(a, b, 0xd7); \
    case 0xd8: return g(a, b, 0xd8); \
    case 0xd9: return g(a, b, 0xd9); \
    case 0xda: return g(a, b, 0xda); \
    case 0xdb: return g(a, b, 0xdb); \
    case 0xdc: return g(a, b, 0xdc); \
    case 0xdd: return g(a, b, 0xdd); \
    case 0xde: return g(a, b, 0xde); \
    case 0xdf: return g(a, b, 0xdf); \
    case 0xe0: return g(a, b, 0xe0); \
    case 0xe1: return g(a, b, 0xe1); \
    case 0xe2: return g(a, b, 0xe2); \
    case 0xe3: return g(a, b, 0xe3); \
    case 0xe4: return g(a, b, 0xe4); \
    case 0xe5: return g(a, b, 0xe5); \
    case 0xe6: return g(a, b, 0xe6); \
    case 0xe7: return g(a, b, 0xe7); \
    case 0xe8: return g(a, b, 0xe8); \
    case 0xe9: return g(a, b, 0xe9); \
    case 0xea: return g(a, b, 0xea); \
    case 0xeb: return g(a, b, 0xeb); \
    case 0xec: return g(a, b, 0xec); \
    case 0xed: return g(a, b, 0xed); \
    case 0xee: return g(a, b, 0xee); \
    case 0xef: return g(a, b, 0xef); \
    case 0xf0: return g(a, b, 0xf0); \
    case 0xf1: return g(a, b, 0xf1); \
    case 0xf2: return g(a, b, 0xf2); \
    case 0xf3: return g(a, b, 0xf3); \
    case 0xf4: return g(a, b, 0xf4); \
    case 0xf5: return g(a, b, 0xf5); \
    case 0xf6: return g(a, b, 0xf6); \
    case 0xf7: return g(a, b, 0xf7); \
    case 0xf8: return g(a, b, 0xf8); \
    case 0xf9: return g(a, b, 0xf9); \
    case 0xfa: return g(a, b, 0xfa); \
    case 0xfb: return g(a, b, 0xfb); \
    case 0xfc: return g(a, b, 0xfc); \
    case 0xfd: return g(a, b, 0xfd); \
    case 0xfe: return g(a, b, 0xfe); \
    default:   return g(a, b, 0xff); \
  } \
}

#define _MM256TWO_LANES_2C(f,g) \
GENERATE_SWITCHED_CALL_2C(g, __m128i, __m128i) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a, const int c) { \
  return _mm256_setr_m128i( \
    g##__switched(_mm256_extractf128_si256(a, 0), c), \
    g##__switched(_mm256_extractf128_si256(a, 1), c) \
  ); \
}

#define _MM256TWO_LANES_3C(f,g) \
GENERATE_SWITCHED_CALL_3C(g, __m128i, __m128i, __m128i) \
__attribute__((always_inline, artificial, const)) \
inline __m256i f (__m256i a, __m256i b, const int c) { \
  return _mm256_setr_m128i( \
    g##__switched(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(a, 0), c), \
    g##__switched(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1), c) \
  ); \
}

#define _mm256_abs_epi8  _Xmm256_abs_epi8
#define _mm256_abs_epi16 _Xmm256_abs_epi16
#define _mm256_abs_epi32 _Xmm256_abs_epi32
_MM256TWO_LANES_1(_mm256_abs_epi8,  _mm_abs_epi8)
_MM256TWO_LANES_1(_mm256_abs_epi16, _mm_abs_epi16)
_MM256TWO_LANES_1(_mm256_abs_epi32, _mm_abs_epi32)

#define _mm256_add_epi8  _Xmm256_add_epi8
#define _mm256_add_epi16 _Xmm256_add_epi16
#define _mm256_add_epi32 _Xmm256_add_epi32
#define _mm256_add_epi64 _Xmm256_add_epi64
_MM256TWO_LANES_2(_mm256_add_epi8,  _mm_add_epi8)
_MM256TWO_LANES_2(_mm256_add_epi16, _mm_add_epi16)
_MM256TWO_LANES_2(_mm256_add_epi32, _mm_add_epi32)
_MM256TWO_LANES_2(_mm256_add_epi64, _mm_add_epi64)

#define _mm256_adds_epi8  _Xmm256_adds_epi8
#define _mm256_adds_epi16 _Xmm256_adds_epi16
#define _mm256_adds_epu8  _Xmm256_adds_epu8
#define _mm256_adds_epu16 _Xmm256_adds_epu16
_MM256TWO_LANES_2(_mm256_adds_epi8,  _mm_adds_epi8)
_MM256TWO_LANES_2(_mm256_adds_epi16, _mm_adds_epi16)
_MM256TWO_LANES_2(_mm256_adds_epu8,  _mm_adds_epu8)
_MM256TWO_LANES_2(_mm256_adds_epu16, _mm_adds_epu16)

#undef _mm256_alignr_epi8
#define _mm256_alignr_epi8 _Xmm256_alignr_epi8
_MM256TWO_LANES_3C(_mm256_alignr_epi8, _mm_alignr_epi8)

#define _mm256_and_si256    _Xmm256_and_si256
#define _mm256_andnot_si256 _Xmm256_andot_si256
_MM256TWO_LANES_2(_mm256_and_si256,    _mm_and_si128)
_MM256TWO_LANES_2(_mm256_andnot_si256, _mm_andnot_si128)

#define _mm256_avg_epu8  _Xmm256_avg_epu8
#define _mm256_avg_epu16 _Xmm256_avg_epu16
_MM256TWO_LANES_2(_mm256_avg_epu8,  _mm_avg_epu8)
_MM256TWO_LANES_2(_mm256_avg_epu16, _mm_avg_epu16)

#undef _mm256_blend_epi16
#define _mm256_blend_epi16 _Xmm256_blend_epi16
_MM256TWO_LANES_3C(_mm256_blend_epi16, _mm_blend_epi16)

GENERATE_SWITCHED_CALL_2C_7(_mm_extract_epi32, int, __m128i)
GENERATE_SWITCHED_CALL_2C_7(_mm256_extract_epi32, int, __m256i)
GENERATE_SWITCHED_CALL_3C_7(_mm256_insert_epi32, __m256i, __m256i, int)

#undef _mm_blend_epi32
#define _mm_blend_epi32 _Xmm_blend_epi32
__attribute__((always_inline, artificial, const))
inline __m128i _mm_blend_epi32 (__m128i a, __m128i b, const int c) {
  int dst[4];
  for (int j = 0; j < 4; j++) {
    if (((c >> j) & 1) == 1) {
      dst[j] = _mm_extract_epi32__switched(b, j);
    } else {
      dst[j] = _mm_extract_epi32__switched(a, j);
    }
  }
  
  return _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
}

#undef _mm256_blend_epi32
#define _mm256_blend_epi32 _Xmm256_blend_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int c) {
  __m256i dst;
  for (int j = 0; j < 8; j++) {
    if (((c >> j) & 1) == 1) {
      dst = _mm256_insert_epi32__switched(dst, _mm256_extract_epi32__switched(b, j), j);
    } else {
      dst = _mm256_insert_epi32__switched(dst, _mm256_extract_epi32__switched(a, j), j);
    }
  }
  
  return dst;
}

#undef _mm256_blendv_epi8
#define _mm256_blendv_epi8 _Xmm256_blendv_epi8
_MM256TWO_LANES_3(_mm256_blendv_epi8, _mm_blendv_epi8)

#define _MM256BROADCASTI(f,g,h,i) \
__attribute__((always_inline, artificial, const)) \
inline __m128i f (__m128i a) { \
  return h(i(a, 0)); \
} \
__attribute__((always_inline, artificial, const)) \
inline __m256i g (__m128i a) { \
  return _mm256_setr_m128i(f(a), f(a)); \
}

#undef _mm_broadcastb_epi8
#undef _mm_broadcastw_epi16
#undef _mm_broadcastd_epi32
#undef _mm_broadcastq_epi64
#define _mm_broadcastb_epi8  _Xmm_broadcastb_epi8
#define _mm_broadcastw_epi16 _Xmm_broadcastw_epi16
#define _mm_broadcastd_epi32 _Xmm_broadcastd_epi32
#define _mm_broadcastq_epi64 _Xmm_broadcastq_epi64
#undef _mm256_broadcastb_epi8
#undef _mm256_broadcastw_epi16
#undef _mm256_broadcastd_epi32
#undef _mm256_broadcastq_epi64
#define _mm256_broadcastb_epi8  _Xmm256_broadcastb_epi8
#define _mm256_broadcastw_epi16 _Xmm256_broadcastw_epi16
#define _mm256_broadcastd_epi32 _Xmm256_broadcastd_epi32
#define _mm256_broadcastq_epi64 _Xmm256_broadcastq_epi64
_MM256BROADCASTI(_mm_broadcastb_epi8,  _mm256_broadcastb_epi8,  _mm_set1_epi8,  _mm_extract_epi8)
_MM256BROADCASTI(_mm_broadcastw_epi16, _mm256_broadcastw_epi16, _mm_set1_epi16, _mm_extract_epi16)
_MM256BROADCASTI(_mm_broadcastd_epi32, _mm256_broadcastd_epi32, _mm_set1_epi32, _mm_extract_epi32)
_MM256BROADCASTI(_mm_broadcastq_epi64, _mm256_broadcastq_epi64, _mm_set1_epi64x, _mm_extract_epi64)

#define _mm_broadcastsd_pd _Xmm_broadcastsd_pd
__attribute__((always_inline, artificial, const))
inline __m128d _mm_broadcastsd_pd (__m128d a) {
  return (__m128d)_mm_broadcastq_epi64((__m128i)a);
}

#define _mm256_broadcastsd_pd _Xmm256_broadcastsd_pd
__attribute__((always_inline, artificial, const))
inline __m256d _mm256_broadcastsd_pd (__m128d a) {
  return (__m256d)_mm256_broadcastq_epi64((__m128i)a);
}

#define _mm256_broadcastsi128_si256 _Xmm256_broadcastsi128_si256
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_broadcastsi128_si256 (__m128i a) {
  return _mm256_setr_m128i(a, a);
}

#define _mm_broadcastss_ps _Xmm_broadcastss_ps
__attribute__((always_inline, artificial, const))
inline __m128d _mm_broadcastss_ps (__m128d a) {
  return (__m128d)_mm_broadcastd_epi32((__m128i)a);
}

#define _mm256_broadcastss_ps _Xmm256_broadcastss_ps
__attribute__((always_inline, artificial, const))
inline __m256d _mm256_broadcastss_ps (__m128d a) {
  return (__m256d)_mm256_broadcastd_epi32((__m128i)a);
}

#define _mm256_cmpeq_epi8  _Xmm256_cmpeq_epi8
#define _mm256_cmpeq_epi16 _Xmm256_cmpeq_epi16
#define _mm256_cmpeq_epi32 _Xmm256_cmpeq_epi32
#define _mm256_cmpeq_epi64 _Xmm256_cmpeq_epi64
_MM256TWO_LANES_2(_mm256_cmpeq_epi8,  _mm_cmpeq_epi8)
_MM256TWO_LANES_2(_mm256_cmpeq_epi16, _mm_cmpeq_epi16)
_MM256TWO_LANES_2(_mm256_cmpeq_epi32, _mm_cmpeq_epi32)
_MM256TWO_LANES_2(_mm256_cmpeq_epi64, _mm_cmpeq_epi64)

#define _mm256_cmpgt_epi8  _Xmm256_cmpgt_epi8
#define _mm256_cmpgt_epi16 _Xmm256_cmpgt_epi16
#define _mm256_cmpgt_epi32 _Xmm256_cmpgt_epi32
#define _mm256_cmpgt_epi64 _Xmm256_cmpgt_epi64
_MM256TWO_LANES_2(_mm256_cmpgt_epi8,  _mm_cmpgt_epi8)
_MM256TWO_LANES_2(_mm256_cmpgt_epi16, _mm_cmpgt_epi16)
_MM256TWO_LANES_2(_mm256_cmpgt_epi32, _mm_cmpgt_epi32)
_MM256TWO_LANES_2(_mm256_cmpgt_epi64, _mm_cmpgt_epi64)

#define _mm256_cvtepi8_epi16  _Xmm256_cvtepi8_epi16
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi8_epi16 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi8_epi16(a),
    _mm_cvtepi8_epi16(_mm_srli_si128(a, 8))
  );
}

#define _mm256_cvtepi8_epi32  _Xmm256_cvtepi8_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi8_epi32 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi8_epi32(a),
    _mm_cvtepi8_epi32(_mm_srli_si128(a, 4))
  );
}

#define _mm256_cvtepi8_epi64  _Xmm256_cvtepi8_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi8_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi8_epi64(a),
    _mm_cvtepi8_epi64(_mm_srli_si128(a, 2))
  );
}

#define _mm256_cvtepi16_epi32  _Xmm256_cvtepi16_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi16_epi32 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi16_epi32(a),
    _mm_cvtepi16_epi32(_mm_srli_si128(a, 8))
  );
}

#define _mm256_cvtepi16_epi64  _Xmm256_cvtepi16_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi16_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi16_epi64(a),
    _mm_cvtepi16_epi64(_mm_srli_si128(a, 4))
  );
}

#define _mm256_cvtepi32_epi64  _Xmm256_cvtepi32_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepi32_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepi32_epi64(a),
    _mm_cvtepi32_epi64(_mm_srli_si128(a, 8))
  );
}

#define _mm256_cvtepu8_epi16  _Xmm256_cvtepu8_epi16
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu8_epi16 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu8_epi16(a),
    _mm_cvtepu8_epi16(_mm_srli_si128(a, 8))
  );
}

#define _mm256_cvtepu8_epi32  _Xmm256_cvtepu8_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu8_epi32 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu8_epi32(a),
    _mm_cvtepu8_epi32(_mm_srli_si128(a, 4))
  );
}

#define _mm256_cvtepu8_epi64  _Xmm256_cvtepu8_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu8_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu8_epi64(a),
    _mm_cvtepu8_epi64(_mm_srli_si128(a, 2))
  );
}

#define _mm256_cvtepu16_epi32  _Xmm256_cvtepu16_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu16_epi32 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu16_epi32(a),
    _mm_cvtepu16_epi32(_mm_srli_si128(a, 8))
  );
}

#define _mm256_cvtepu16_epi64  _Xmm256_cvtepu16_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu16_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu16_epi64(a),
    _mm_cvtepu16_epi64(_mm_srli_si128(a, 4))
  );
}

#define _mm256_cvtepu32_epi64  _Xmm256_cvtepu32_epi64
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_cvtepu32_epi64 (__m128i a) {
  return _mm256_setr_m128i(
    _mm_cvtepu32_epi64(a),
    _mm_cvtepu32_epi64(_mm_srli_si128(a, 8))
  );
}

#undef _mm256_extracti128_si256
#define _mm256_extracti128_si256 _mm256_extractf128_si256

#define _mm256_hadd_epi16  _Xmm256_hadd_epi16
#define _mm256_hadd_epi32  _Xmm256_hadd_epi32
#define _mm256_hadds_epi16 _Xmm256_hadds_epi16
_MM256TWO_LANES_2(_mm256_hadd_epi16, _mm_hadd_epi16)
_MM256TWO_LANES_2(_mm256_hadd_epi32, _mm_hadd_epi32)
_MM256TWO_LANES_2(_mm256_hadds_epi16, _mm_hadds_epi16)

#define _mm256_hsub_epi16  _Xmm256_hsub_epi16
#define _mm256_hsub_epi32  _Xmm256_hsub_epi32
#define _mm256_hsubs_epi16 _Xmm256_hsubs_epi16
_MM256TWO_LANES_2(_mm256_hsub_epi16, _mm_hsub_epi16)
_MM256TWO_LANES_2(_mm256_hsub_epi32, _mm_hsub_epi32)
_MM256TWO_LANES_2(_mm256_hsubs_epi16, _mm_hsubs_epi16)

#define _MM256GATHER(f,fm,g,gm,t,e,q,p,n) \
__attribute__((always_inline, artificial)) \
inline __m128i f (t const* base_addr, __m128i vindex, const int scale) { \
  t dst[n]; \
  for (int j = 0; j < n; j++) { \
    dst[j] = *(t*)(base_addr + q(vindex, j) * scale); \
  } \
  return _mm_loadu_si128((const __m128i*)&dst[0]); \
} \
__attribute__((always_inline, artificial)) \
inline __m128i fm (__m128i src, t const* base_addr, __m128i vindex, __m128i mask, const int scale) { \
  t dst[n]; \
  for (int j = 0; j < n; j++) { \
    if (_mm_extract_epi32(mask, j) < 0) { \
      dst[j] = *(t*)(base_addr + q(vindex, j) * scale); \
    } else { \
      dst[j] = p(src, j); \
    } \
  } \
  \
  return _mm_loadu_si128((const __m128i*)&dst[0]); \
} \
__attribute__((always_inline, artificial)) \
inline __m256i g (t const* base_addr, __m256i vindex, const int scale) { \
  return _mm256_setr_m128i( \
    f(base_addr, _mm256_extractf128_si256(vindex, 0), scale), \
    f(base_addr, _mm256_extractf128_si256(vindex, 1), scale) \
  ); \
} \
__attribute__((always_inline, artificial)) \
inline __m256i gm (__m256i src, t const* base_addr, __m256i vindex, __m256i mask, const int scale) { \
  return _mm256_setr_m128i( \
    fm(_mm256_extractf128_si256(src, 0), base_addr, _mm256_extractf128_si256(vindex, 0), _mm256_extractf128_si256(mask, 0), scale), \
    fm(_mm256_extractf128_si256(src, 1), base_addr, _mm256_extractf128_si256(vindex, 1), _mm256_extractf128_si256(mask, 1), scale) \
  ); \
}

#undef _mm_i32gather_epi32
#undef _mm_i32gather_epi64
#undef _mm_i64gather_epi32
#undef _mm_i64gather_epi64
#define _mm_i32gather_epi32 _Xmm_i32gather_epi32
#define _mm_i32gather_epi64 _Xmm_i32gather_epi64
#define _mm_i64gather_epi32 _Xmm_i64gather_epi32
#define _mm_i64gather_epi64 _Xmm_i64gather_epi64
#undef _mm256_i32gather_epi32
#undef _mm256_i32gather_epi64
#undef _mm256_i64gather_epi32
#undef _mm256_i64gather_epi64
#define _mm256_i32gather_epi32 _Xmm256_i32gather_epi32
#define _mm256_i32gather_epi64 _Xmm256_i32gather_epi64
#define _mm256_i64gather_epi32 _Xmm256_i64gather_epi32
#define _mm256_i64gather_epi64 _Xmm256_i64gather_epi64
#undef _mm_mask_i32gather_epi32
#undef _mm_mask_i32gather_epi64
#undef _mm_mask_i64gather_epi32
#undef _mm_mask_i64gather_epi64
#define _mm_mask_i32gather_epi32 _Xmm_mask_i32gather_epi32
#define _mm_mask_i32gather_epi64 _Xmm_mask_i32gather_epi64
#define _mm_mask_i64gather_epi32 _Xmm_mask_i64gather_epi32
#define _mm_mask_i64gather_epi64 _Xmm_mask_i64gather_epi64
#undef _mm256_mask_i32gather_epi32
#undef _mm256_mask_i32gather_epi64
#undef _mm256_mask_i64gather_epi32
#undef _mm256_mask_i64gather_epi64
#define _mm256_mask_i32gather_epi32 _Xmm256_mask_i32gather_epi32
#define _mm256_mask_i32gather_epi64 _Xmm256_mask_i32gather_epi64
#define _mm256_mask_i64gather_epi32 _Xmm256_mask_i64gather_epi32
#define _mm256_mask_i64gather_epi64 _Xmm256_mask_i64gather_epi64
_MM256GATHER(_mm_i32gather_epi32, _mm_mask_i32gather_epi32, _mm256_i32gather_epi32, _mm256_mask_i32gather_epi32, int, i32, _mm_extract_epi32, _mm_extract_epi32, 4)
_MM256GATHER(_mm_i32gather_epi64, _mm_mask_i32gather_epi64, _mm256_i32gather_epi64, _mm256_mask_i32gather_epi64, __int64, i64, _mm_extract_epi32, _mm_extract_epi64, 2)
_MM256GATHER(_mm_i64gather_epi32, _mm_mask_i64gather_epi32, _mm256_i64gather_epi32, _mm256_mask_i64gather_epi32, int, i32, _mm_extract_epi64, _mm_extract_epi32, 2)
_MM256GATHER(_mm_i64gather_epi64, _mm_mask_i64gather_epi64, _mm256_i64gather_epi64, _mm256_mask_i64gather_epi64, __int64, i64, _mm_extract_epi64, _mm_extract_epi64, 2)

#undef _mm_i32gather_pd
#define _mm_i32gather_pd _Xmm_i32gather_pd
__attribute__((always_inline, artificial))
inline __m128d _mm_i32gather_pd (double const* base_addr, __m128i vindex, const int scale) {
  return (__m128d)_mm_i32gather_epi64((__int64 const*) base_addr, vindex, scale);
}

#undef _mm_mask_i32gather_pd
#define _mm_mask_i32gather_pd _Xmm_mask_i32gather_pd
__attribute__((always_inline, artificial))
inline __m128d _mm_mask_i32gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale) {
  return (__m128d)_mm_mask_i32gather_epi64((__m128i) src, (__int64 const*) base_addr, vindex, (__m128i) mask, scale);
}

#undef _mm256_i32gather_pd
#define _mm256_i32gather_pd _Xmm256_i32gather_pd
__attribute__((always_inline, artificial))
inline __m256d _mm256_i32gather_pd (double const* base_addr, __m256i vindex, const int scale) {
  return (__m256d)_mm256_i32gather_epi64((__int64 const*) base_addr, vindex, scale);
}

#undef _mm256_mask_i32gather_pd
#define _mm256_mask_i32gather_pd _Xmm256_mask_i32gather_pd
__attribute__((always_inline, artificial))
inline __m256d _mm256_mask_i32gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale) {
  return (__m256d)_mm256_mask_i32gather_epi64((__m256i) src, (__int64 const*) base_addr, vindex, (__m256i) mask, scale);
}

#undef _mm_i32gather_ps
#define _mm_i32gather_ps _Xmm_i32gather_ps
__attribute__((always_inline, artificial))
inline __m128d _mm_i32gather_ps (float const* base_addr, __m128i vindex, const int scale) {
  return (__m128d)_mm_i32gather_epi32((int const*) base_addr, vindex, scale);
}

#undef _mm_mask_i32gather_ps
#define _mm_mask_i32gather_ps _Xmm_mask_i32gather_ps
__attribute__((always_inline, artificial))
inline __m128d _mm_mask_i32gather_ps (__m128d src, float const* base_addr, __m128i vindex, __m128d mask, const int scale) {
  return (__m128d)_mm_mask_i32gather_epi32((__m128i) src, (int const*) base_addr, vindex, (__m128i) mask, scale);
}

#undef _mm256_i32gather_ps
#define _mm256_i32gather_ps _Xmm256_i32gather_ps
__attribute__((always_inline, artificial))
inline __m256d _mm256_i32gather_ps (float const* base_addr, __m256i vindex, const int scale) {
  return (__m256d)_mm256_i32gather_epi32((int const*) base_addr, vindex, scale);
}

#undef _mm256_mask_i32gather_ps
#define _mm256_mask_i32gather_ps _Xmm256_mask_i32gather_ps
__attribute__((always_inline, artificial))
inline __m256d _mm256_mask_i32gather_ps (__m256d src, float const* base_addr, __m256i vindex, __m256d mask, const int scale) {
  return (__m256d)_mm256_mask_i32gather_epi32((__m256i) src, (int const*) base_addr, vindex, (__m256i) mask, scale);
}

#undef _mm_i64gather_pd
#define _mm_i64gather_pd _Xmm_i64gather_pd
__attribute__((always_inline, artificial))
inline __m128d _mm_i64gather_pd (double const* base_addr, __m128i vindex, const int scale) {
  return (__m128d)_mm_i64gather_epi64((__int64 const*) base_addr, vindex, scale);
}

#undef _mm_mask_i64gather_pd
#define _mm_mask_i64gather_pd _Xmm_mask_i64gather_pd
__attribute__((always_inline, artificial))
inline __m128d _mm_mask_i64gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale) {
  return (__m128d)_mm_mask_i64gather_epi64((__m128i) src, (__int64 const*) base_addr, vindex, (__m128i) mask, scale);
}

#undef _mm256_i64gather_pd
#define _mm256_i64gather_pd _Xmm256_i64gather_pd
__attribute__((always_inline, artificial))
inline __m256d _mm256_i64gather_pd (double const* base_addr, __m256i vindex, const int scale) {
  return (__m256d)_mm256_i64gather_epi64((__int64 const*) base_addr, vindex, scale);
}

#undef _mm256_mask_i64gather_pd
#define _mm256_mask_i64gather_pd _Xmm256_mask_i64gather_pd
__attribute__((always_inline, artificial))
inline __m256d _mm256_mask_i64gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale) {
  return (__m256d)_mm256_mask_i64gather_epi64((__m256i) src, (__int64 const*) base_addr, vindex, (__m256i) mask, scale);
}

#undef _mm_i64gather_ps
#define _mm_i64gather_ps _Xmm_i64gather_ps
__attribute__((always_inline, artificial))
inline __m128d _mm_i64gather_ps (float const* base_addr, __m128i vindex, const int scale) {
  return (__m128d)_mm_i64gather_epi32((int const*) base_addr, vindex, scale);
}

#undef _mm_mask_i64gather_ps
#define _mm_mask_i64gather_ps _Xmm_mask_i64gather_ps
__attribute__((always_inline, artificial))
inline __m128d _mm_mask_i64gather_ps (__m128d src, float const* base_addr, __m128i vindex, __m128d mask, const int scale) {
  return (__m128d)_mm_mask_i64gather_epi32((__m128i) src, (int const*) base_addr, vindex, (__m128i) mask, scale);
}

#undef _mm256_i64gather_ps
#define _mm256_i64gather_ps _Xmm256_i64gather_ps
__attribute__((always_inline, artificial))
inline __m256d _mm256_i64gather_ps (float const* base_addr, __m256i vindex, const int scale) {
  return (__m256d)_mm256_i64gather_epi32((int const*) base_addr, vindex, scale);
}

#undef _mm256_mask_i64gather_ps
#define _mm256_mask_i64gather_ps _Xmm256_mask_i64gather_ps
__attribute__((always_inline, artificial))
inline __m256d _mm256_mask_i64gather_ps (__m256d src, float const* base_addr, __m256i vindex, __m256d mask, const int scale) {
  return (__m256d)_mm256_mask_i64gather_epi32((__m256i) src, (int const*) base_addr, vindex, (__m256i) mask, scale);
}

#undef _mm256_inserti128_si256
#define _mm256_inserti128_si256 _Xmm256_inserti128_si256
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_inserti128_si256 (__m256i a, __m128i b, const int imm8) {
  if ((imm8 & 1) == 0) {
    return _mm256_setr_m128i(b, _mm256_extractf128_si256(a, 1));
  } else {
    return _mm256_setr_m128i(_mm256_extractf128_si256(a, 0), b);
  }
}

#define _mm256_madd_epi16 _Xmm256_madd_epi16
#define _mm256_maddubs_epi16 _Xmm256_maddubs_epi16
_MM256TWO_LANES_2(_mm256_madd_epi16, _mm_madd_epi16)
_MM256TWO_LANES_2(_mm256_maddubs_epi16, _mm_maddubs_epi16)

#define _mm_maskload_epi32 _Xmm_maskload_epi32
__attribute__((always_inline, artificial))
inline __m128i _mm_maskload_epi32 (int const* mem_addr, __m128i mask) {
  int dst[4];
  for (int j = 0; j < 4; j++) {
    if (_mm_extract_epi32(mask, j) < 0) {
      dst[j] = mem_addr[j];
    } else {
      dst[j] = 0;
    }
  }
  
  return _mm_setr_epi32(dst[0], dst[1], dst[2], dst[3]);
}

#define _mm256_maskload_epi32 _Xmm256_maskload_epi32
__attribute__((always_inline, artificial))
inline __m256i _mm256_maskload_epi32 (int const* mem_addr, __m256i mask) {
  return _mm256_setr_m128i(
    _mm_maskload_epi32(mem_addr, _mm256_extractf128_si256(mask, 0)),
    _mm_maskload_epi32(mem_addr + 4, _mm256_extractf128_si256(mask, 1))
  );
}

#define _mm_maskload_epi64 _Xmm_maskload_epi64
__attribute__((always_inline, artificial))
inline __m128i _mm_maskload_epi64 (__int64 const* mem_addr, __m128i mask) {
  __int64 dst[2];
  for (int j = 0; j < 2; j++) {
    if (_mm_extract_epi64(mask, j) < 0) {
      dst[j] = mem_addr[j];
    } else {
      dst[j] = 0;
    }
  }
  
  return _mm_set_epi64x(dst[1], dst[0]);
}

#define _mm256_maskload_epi64 _Xmm256_maskload_epi64
__attribute__((always_inline, artificial))
inline __m256i _mm256_maskload_epi64 (__int64 const* mem_addr, __m256i mask) {
  return _mm256_setr_m128i(
    _mm_maskload_epi64(mem_addr, _mm256_extractf128_si256(mask, 0)),
    _mm_maskload_epi64(mem_addr + 2, _mm256_extractf128_si256(mask, 1))
  );
}

#define _mm_maskstore_epi32 _Xmm_maskstore_epi32
__attribute__((always_inline, artificial))
inline void _mm_maskstore_epi32 (int* mem_addr, __m128i mask, __m128i a) {
  for (int j = 0; j < 4; j++) {
    if (_mm_extract_epi32(mask, j) < 0) {
      mem_addr[j] = _mm_extract_epi32(a, j);
    }
  }
}

#define _mm256_maskstore_epi32 _Xmm256_maskstore_epi32
__attribute__((always_inline, artificial))
inline void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a) {
  _mm_maskstore_epi32(mem_addr, _mm256_extractf128_si256(mask, 0), _mm256_extractf128_si256(a, 0));
  _mm_maskstore_epi32(mem_addr + 4, _mm256_extractf128_si256(mask, 1), _mm256_extractf128_si256(a, 1));
}

#define _mm_maskstore_epi64 _Xmm_maskstore_epi64
__attribute__((always_inline, artificial))
inline void _mm_maskstore_epi64 (__int64* mem_addr, __m128i mask, __m128i a) {
  for (int j = 0; j < 2; j++) {
    if (_mm_extract_epi64(mask, j) < 0) {
      mem_addr[j] = _mm_extract_epi64(a, j);
    }
  }
}

#define _mm256_maskstore_epi64 _Xmm256_maskstore_epi64
__attribute__((always_inline, artificial))
inline void _mm256_maskstore_epi64 (__int64* mem_addr, __m256i mask, __m256i a) {
  _mm_maskstore_epi64(mem_addr, _mm256_extractf128_si256(mask, 0), _mm256_extractf128_si256(a, 0));
  _mm_maskstore_epi64(mem_addr + 2, _mm256_extractf128_si256(mask, 1), _mm256_extractf128_si256(a, 1));
}

#define _mm256_max_epi8  _Xmm256_max_epi8
#define _mm256_max_epi16 _Xmm256_max_epi16
#define _mm256_max_epi32 _Xmm256_max_epi32
_MM256TWO_LANES_2(_mm256_max_epi8,  _mm_max_epi8)
_MM256TWO_LANES_2(_mm256_max_epi16, _mm_max_epi16)
_MM256TWO_LANES_2(_mm256_max_epi32, _mm_max_epi32)

#define _mm256_max_epu8  _Xmm256_max_epu8
#define _mm256_max_epu16 _Xmm256_max_epu16
#define _mm256_max_epu32 _Xmm256_max_epu32
_MM256TWO_LANES_2(_mm256_max_epu8,  _mm_max_epu8)
_MM256TWO_LANES_2(_mm256_max_epu16, _mm_max_epu16)
_MM256TWO_LANES_2(_mm256_max_epu32, _mm_max_epu32)

#define _mm256_min_epi8  _Xmm256_min_epi8
#define _mm256_min_epi16 _Xmm256_min_epi16
#define _mm256_min_epi32 _Xmm256_min_epi32
_MM256TWO_LANES_2(_mm256_min_epi8,  _mm_min_epi8)
_MM256TWO_LANES_2(_mm256_min_epi16, _mm_min_epi16)
_MM256TWO_LANES_2(_mm256_min_epi32, _mm_min_epi32)

#define _mm256_min_epu8  _Xmm256_min_epu8
#define _mm256_min_epu16 _Xmm256_min_epu16
#define _mm256_min_epu32 _Xmm256_min_epu32
_MM256TWO_LANES_2(_mm256_min_epu8,  _mm_min_epu8)
_MM256TWO_LANES_2(_mm256_min_epu16, _mm_min_epu16)
_MM256TWO_LANES_2(_mm256_min_epu32, _mm_min_epu32)

#define _mm256_movemask_epi8 _Xmm256_movemask_epi8
__attribute__((always_inline, artificial, const))
inline int _mm256_movemask_epi8 (__m256i a) {
  return _mm_movemask_epi8(_mm256_extractf128_si256(a, 0)) | (_mm_movemask_epi8(_mm256_extractf128_si256(a, 1)) << 16);
}

#undef _mm256_mpsadbw_epu8
#define _mm256_mpsadbw_epu8 _Xmm256_mpsadbw_epu8
GENERATE_SWITCHED_CALL_3C(_mm_mpsadbw_epu8, __m128i, __m128i, __m128i)
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_mpsadbw_epu8 (__m256i a, __m256i b, const int imm8) {
  return _mm256_setr_m128i(
    _mm_mpsadbw_epu8__switched(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0), imm8),
    _mm_mpsadbw_epu8__switched(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0), imm8 >> 3)
  );
}

#define _mm256_mul_epi32 _Xmm256_mul_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_mul_epi32 (__m256i a, __m256i b) {
  return _mm256_setr_epi64x(
    _mm256_extract_epi32(a, 0) * _mm256_extract_epi32(b, 0),
    _mm256_extract_epi32(a, 1) * _mm256_extract_epi32(b, 1),
    _mm256_extract_epi32(a, 2) * _mm256_extract_epi32(b, 2),
    _mm256_extract_epi32(a, 3) * _mm256_extract_epi32(b, 3)
  );
}

#define _mm256_mul_epu32 _Xmm256_mul_epu32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_mul_epu32 (__m256i a, __m256i b) {
  return _mm256_setr_epi64x(
    (uint32_t)_mm256_extract_epi32(a, 0) * (uint32_t)_mm256_extract_epi32(b, 0),
    (uint32_t)_mm256_extract_epi32(a, 1) * (uint32_t)_mm256_extract_epi32(b, 1),
    (uint32_t)_mm256_extract_epi32(a, 2) * (uint32_t)_mm256_extract_epi32(b, 2),
    (uint32_t)_mm256_extract_epi32(a, 3) * (uint32_t)_mm256_extract_epi32(b, 3)
  );
}

#define _mm256_mulhi_epi16  _Xmm256_mulhi_epi16
#define _mm256_mullo_epi16  _Xmm256_mullo_epi16
#define _mm256_mulhrs_epi16 _Xmm256_mulhrs_epi16
_MM256TWO_LANES_2(_mm256_mulhi_epi16,  _mm_mulhi_epi16)
_MM256TWO_LANES_2(_mm256_mullo_epi16,  _mm_mullo_epi16)
_MM256TWO_LANES_2(_mm256_mulhrs_epi16, _mm_mulhrs_epi16)

#define _mm256_or_si256 _Xmm256_or_si256
_MM256TWO_LANES_2(_mm256_or_si256, _mm_or_si128)

#define _mm256_packs_epi16  _Xmm256_packs_epi16
#define _mm256_packs_epi32  _Xmm256_packs_epi32
#define _mm256_packus_epi16 _Xmm256_packus_epi16
#define _mm256_packus_epi32 _Xmm256_packus_epi32
_MM256TWO_LANES_2(_mm256_packs_epi16,  _mm_packs_epi16)
_MM256TWO_LANES_2(_mm256_packs_epi32,  _mm_packs_epi32)
_MM256TWO_LANES_2(_mm256_packus_epi16, _mm_packus_epi16)
_MM256TWO_LANES_2(_mm256_packus_epi32, _mm_packus_epi32)

#undef _mm256_permute2x128_si256
#define _mm256_permute2x128_si256 _Xmm256_permute2x128_si256
GENERATE_SWITCHED_CALL_2C_1(_mm256_extracti128_si256, __m128i, __m256i)
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8) {
  return _mm256_setr_m128i(
    imm8 & (1 << 3) ? _mm_setzero_si128() : imm8 & (1 << 1) ?
      _mm256_extracti128_si256__switched(b, imm8 & 1) : _mm256_extracti128_si256__switched(a, imm8 & 1),
    imm8 & (1 << 7) ? _mm_setzero_si128() : imm8 & (1 << 5) ?
      _mm256_extracti128_si256__switched(b, (imm8 >> 4) & 1) : _mm256_extracti128_si256__switched(a, (imm8 >> 4) & 1)
  );
}

#undef _mm256_permute4x64_epi64
#define _mm256_permute4x64_epi64 _Xmm256_permute4x64_epi64
GENERATE_SWITCHED_CALL_2C_3(_mm256_extract_epi64, __int64, __m256i)
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8) {
  return _mm256_setr_epi64x(
    _mm256_extract_epi64__switched(a, (imm8 >> 0) & 3),
    _mm256_extract_epi64__switched(a, (imm8 >> 2) & 3),
    _mm256_extract_epi64__switched(a, (imm8 >> 4) & 3),
    _mm256_extract_epi64__switched(a, (imm8 >> 6) & 3)
  );
}

#undef _mm256_permute4x64_pd
#define _mm256_permute4x64_pd _Xmm256_permute4x64_pd
__attribute__((always_inline, artificial, const))
inline __m256d _mm256_permute4x64_pd (__m256d a, const int imm8) {
  return (__m256d)_mm256_permute4x64_epi64((__m256i)a, imm8);
}

#undef _mm256_permutevar8x32_epi32
#define _mm256_permutevar8x32_epi32 _Xmm256_permutevar8x32_epi32
__attribute__((always_inline, artificial, const))
inline __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx) {
  return _mm256_setr_epi32(
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 0)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 1)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 2)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 3)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 4)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 5)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 6)),
    _mm256_extract_epi32__switched(a, _mm256_extract_epi32(idx, 7))
  );
}

#undef _mm256_permutevar8x32_ps
#define _mm256_permutevar8x32_ps _Xmm256_permutevar8x32_ps
__attribute__((always_inline, artificial, const))
inline __m256 _mm256_permutevar8x32_ps (__m256 a, __m256i idx) {
  return (__m256)_mm256_permutevar8x32_epi32((__m256i)a, idx);
}

#define _mm256_sad_epu8 _Xmm256_sad_epu8
_MM256TWO_LANES_2(_mm256_sad_epu8, _mm_sad_epu8)

#undef _mm256_shuffle_epi32
#define _mm256_shuffle_epi32 _Xmm256_shuffle_epi32
_MM256TWO_LANES_2C(_mm256_shuffle_epi32, _mm_shuffle_epi32)

#undef _mm256_shuffle_epi8
#define _mm256_shuffle_epi8 _Xmm256_shuffle_epi8
_MM256TWO_LANES_2(_mm256_shuffle_epi8, _mm_shuffle_epi8)

#undef _mm256_shufflehi_epi16
#undef _mm256_shufflelo_epi16
#define _mm256_shufflehi_epi16 _Xmm256_shufflehi_epi16
#define _mm256_shufflelo_epi16 _Xmm256_shufflelo_epi16
_MM256TWO_LANES_2C(_mm256_shufflehi_epi16, _mm_shufflehi_epi16)
_MM256TWO_LANES_2C(_mm256_shufflelo_epi16, _mm_shufflelo_epi16)

#define _mm256_sign_epi8  _Xmm256_sign_epi8
#define _mm256_sign_epi16 _Xmm256_sign_epi16
#define _mm256_sign_epi32 _Xmm256_sign_epi32
_MM256TWO_LANES_2(_mm256_sign_epi8,  _mm_sign_epi8)
_MM256TWO_LANES_2(_mm256_sign_epi16, _mm_sign_epi16)
_MM256TWO_LANES_2(_mm256_sign_epi32, _mm_sign_epi32)

#define _mm256_sll_epi16 _Xmm256_sll_epi16
#define _mm256_sll_epi32 _Xmm256_sll_epi32
#define _mm256_sll_epi64 _Xmm256_sll_epi64
_MM256TWO_LANES_2CSI128(_mm256_sll_epi16, _mm_sll_epi16)
_MM256TWO_LANES_2CSI128(_mm256_sll_epi32, _mm_sll_epi32)
_MM256TWO_LANES_2CSI128(_mm256_sll_epi64, _mm_sll_epi64)

#undef _mm256_slli_si256
#define _mm256_slli_epi16 _Xmm256_slli_epi16
#define _mm256_slli_epi32 _Xmm256_slli_epi32
#define _mm256_slli_epi64 _Xmm256_slli_epi64
#define _mm256_slli_si256 _Xmm256_slli_si256
_MM256TWO_LANES_2C(_mm256_slli_epi16, _mm_slli_epi16)
_MM256TWO_LANES_2C(_mm256_slli_epi32, _mm_slli_epi32)
_MM256TWO_LANES_2C(_mm256_slli_epi64, _mm_slli_epi64)
_MM256TWO_LANES_2C(_mm256_slli_si256, _mm_slli_si128)

#define _mm_sllv_epi32 _Xmm_sllv_epi32
__attribute__((always_inline, artificial, const))
inline __m128i _mm_sllv_epi32 (__m128i a, __m128i count) {
  return _mm_setr_epi32(
    _mm_extract_epi32(a, 0) << _mm_extract_epi32(count, 0),
    _mm_extract_epi32(a, 1) << _mm_extract_epi32(count, 1),
    _mm_extract_epi32(a, 2) << _mm_extract_epi32(count, 2),
    _mm_extract_epi32(a, 3) << _mm_extract_epi32(count, 3)
  );
}

#define _mm256_sllv_epi32 _Xmm256_sllv_epi32
_MM256TWO_LANES_2(_mm256_sllv_epi32, _mm_sllv_epi32)

#define _mm_sllv_epi64 _Xmm_sllv_epi64
__attribute__((always_inline, artificial, const))
inline __m128i _mm_sllv_epi64 (__m128i a, __m128i count) {
  return _mm_set_epi64x(
    _mm_extract_epi64(a, 1) << _mm_extract_epi64(count, 1),
    _mm_extract_epi64(a, 0) << _mm_extract_epi64(count, 0)
  );
}

#define _mm256_sllv_epi64 _Xmm256_sllv_epi64
_MM256TWO_LANES_2(_mm256_sllv_epi64, _mm_sllv_epi64)

#define _mm256_sra_epi16 _Xmm256_sra_epi16
#define _mm256_sra_epi32 _Xmm256_sra_epi32
_MM256TWO_LANES_2(_mm256_sra_epi16, _mm_sra_epi16)
_MM256TWO_LANES_2(_mm256_sra_epi32, _mm_sra_epi32)

#define _mm256_srai_epi16 _Xmm256_srai_epi16
#define _mm256_srai_epi32 _Xmm256_srai_epi32
_MM256TWO_LANES_2C(_mm256_srai_epi16, _mm_srai_epi16)
_MM256TWO_LANES_2C(_mm256_srai_epi32, _mm_srai_epi32)

#define _mm_srav_epi32 _Xmm_srav_epi32
__attribute__((always_inline, artificial, const))
inline __m128i _mm_srav_epi32 (__m128i a, __m128i count) {
  return _mm_setr_epi32(
    _mm_extract_epi32(a, 0) >> _mm_extract_epi32(count, 0),
    _mm_extract_epi32(a, 1) >> _mm_extract_epi32(count, 1),
    _mm_extract_epi32(a, 2) >> _mm_extract_epi32(count, 2),
    _mm_extract_epi32(a, 3) >> _mm_extract_epi32(count, 3)
  );
}

#define _mm256_srav_epi32 _Xmm256_srav_epi32
_MM256TWO_LANES_2(_mm256_srav_epi32, _mm_srav_epi32)

#define _mm256_srl_epi16 _Xmm256_srl_epi16
#define _mm256_srl_epi32 _Xmm256_srl_epi32
_MM256TWO_LANES_2(_mm256_srl_epi16, _mm_srl_epi16)
_MM256TWO_LANES_2(_mm256_srl_epi32, _mm_srl_epi32)

#define _mm256_srli_epi16 _Xmm256_srli_epi16
#define _mm256_srli_epi32 _Xmm256_srli_epi32
_MM256TWO_LANES_2C(_mm256_srli_epi16, _mm_srli_epi16)
_MM256TWO_LANES_2C(_mm256_srli_epi32, _mm_srli_epi32)

#define _mm_srlv_epi32 _Xmm_srlv_epi32
__attribute__((always_inline, artificial, const))
inline __m128i _mm_srlv_epi32 (__m128i a, __m128i count) {
  return _mm_setr_epi32(
    (uint32_t)_mm_extract_epi32(a, 0) >> _mm_extract_epi32(count, 0),
    (uint32_t)_mm_extract_epi32(a, 1) >> _mm_extract_epi32(count, 1),
    (uint32_t)_mm_extract_epi32(a, 2) >> _mm_extract_epi32(count, 2),
    (uint32_t)_mm_extract_epi32(a, 3) >> _mm_extract_epi32(count, 3)
  );
}

#define _mm256_srlv_epi32 _Xmm256_srlv_epi32
_MM256TWO_LANES_2(_mm256_srlv_epi32, _mm_srlv_epi32)

#undef _mm256_bslli_epi128
#undef _mm256_bsrli_epi128
#define _mm256_bslli_epi128 _mm256_slli_si256
#define _mm256_bsrli_epi128 _mm256_srli_si256

#define _mm256_stream_load_si256 _Xmm256_stream_load_si256
__attribute__((always_inline, artificial))
inline __m256i _mm256_stream_load_si256 (__m256i const* mem_addr) {
  return _mm256_loadu_si256(mem_addr);
}

#define _mm256_sub_epi8  _Xmm256_sub_epi8
#define _mm256_sub_epi16 _Xmm256_sub_epi16
#define _mm256_sub_epi32 _Xmm256_sub_epi32
#define _mm256_sub_epi64 _Xmm256_sub_epi64
_MM256TWO_LANES_2(_mm256_sub_epi8,  _mm_sub_epi8)
_MM256TWO_LANES_2(_mm256_sub_epi16, _mm_sub_epi16)
_MM256TWO_LANES_2(_mm256_sub_epi32, _mm_sub_epi32)
_MM256TWO_LANES_2(_mm256_sub_epi64, _mm_sub_epi64)

#define _mm256_subs_epi8  _Xmm256_subs_epi8
#define _mm256_subs_epi16 _Xmm256_subs_epi16
#define _mm256_subs_epu8  _Xmm256_subs_epu8
#define _mm256_subs_epu16 _Xmm256_subs_epu16
_MM256TWO_LANES_2(_mm256_subs_epi8,  _mm_subs_epi8)
_MM256TWO_LANES_2(_mm256_subs_epi16, _mm_subs_epi16)
_MM256TWO_LANES_2(_mm256_subs_epu8,  _mm_subs_epu8)
_MM256TWO_LANES_2(_mm256_subs_epu16, _mm_subs_epu16)

#define _mm256_unpackhi_epi8  _Xmm256_unpackhi_epi8
#define _mm256_unpackhi_epi16 _Xmm256_unpackhi_epi16
#define _mm256_unpackhi_epi32 _Xmm256_unpackhi_epi32
#define _mm256_unpackhi_epi64 _Xmm256_unpackhi_epi64
_MM256TWO_LANES_2(_mm256_unpackhi_epi8,  _mm_unpackhi_epi8)
_MM256TWO_LANES_2(_mm256_unpackhi_epi16, _mm_unpackhi_epi16)
_MM256TWO_LANES_2(_mm256_unpackhi_epi32, _mm_unpackhi_epi32)
_MM256TWO_LANES_2(_mm256_unpackhi_epi64, _mm_unpackhi_epi64)

#define _mm256_unpacklo_epi8  _Xmm256_unpacklo_epi8
#define _mm256_unpacklo_epi16 _Xmm256_unpacklo_epi16
#define _mm256_unpacklo_epi32 _Xmm256_unpacklo_epi32
#define _mm256_unpacklo_epi64 _Xmm256_unpacklo_epi64
_MM256TWO_LANES_2(_mm256_unpacklo_epi8,  _mm_unpacklo_epi8)
_MM256TWO_LANES_2(_mm256_unpacklo_epi16, _mm_unpacklo_epi16)
_MM256TWO_LANES_2(_mm256_unpacklo_epi32, _mm_unpacklo_epi32)
_MM256TWO_LANES_2(_mm256_unpacklo_epi64, _mm_unpacklo_epi64)

#define _mm256_xor_si256 _Xmm256_xor_si256
_MM256TWO_LANES_2(_mm256_xor_si256, _mm_xor_si128)

#endif /* __AVX2__ */

#endif /* _SW_AVX2INTRIN_H_INCLUDED */

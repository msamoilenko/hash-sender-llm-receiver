/* 32-bit tag: keyed BLAKE2b, 4-byte digest, big-endian (assim.hashing.tag). Every method sends it. */
#include <stdint.h>

#include "blake2.h"

uint32_t tag32(const uint8_t *msg, uint8_t len, const uint8_t *key, uint8_t keylen) {
    uint8_t d[4];
    blake2b(d, sizeof d, msg, len, key, keylen);
    return ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) | ((uint32_t)d[2] << 8) | d[3];
}

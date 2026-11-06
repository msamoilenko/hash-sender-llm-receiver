/* Hash-only sender of the `10:4` scheme (assim.emitter.encode): chunk_bytes 2, head_bytes 8,
 * 10 bits on chunks starting in the first 8 bytes and 4 bits elsewhere. Each chunk hash is keyed
 * BLAKE2b ("assim-chunk-v1") over index (2 bytes, big-endian) + whole prefix, 8-byte digest,
 * top `bits` bits. The prefix is rehashed from scratch per chunk: minimal state, time not measured.
 * Returns the number of chunks. */
#include <stdint.h>

#include "blake2.h"

#define CHUNK_BYTES 2
#define HEAD_BYTES 8
#define HEAD_BITS 10
#define TAIL_BITS 4

static const uint8_t CHUNK_KEY[] = {'a', 's', 's', 'i', 'm', '-', 'c', 'h', 'u', 'n', 'k', '-', 'v', '1'};

uint32_t scheme_encode(const uint8_t *msg, uint8_t len, uint16_t *hashes) {
    uint32_t n = 0;
    for (uint32_t start = 0; start < len; start += CHUNK_BYTES, n++) {
        uint32_t end = start + CHUNK_BYTES < len ? start + CHUNK_BYTES : len;
        uint32_t bits = start < HEAD_BYTES ? HEAD_BITS : TAIL_BITS;
        uint8_t idx[2] = {(uint8_t)(n >> 8), (uint8_t)n}, d[8];
        blake2b_state s;
        blake2b_init_key(&s, sizeof d, CHUNK_KEY, sizeof CHUNK_KEY);
        blake2b_update(&s, idx, sizeof idx);
        blake2b_update(&s, msg, end);
        blake2b_final(&s, d, sizeof d);
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | d[j];
        hashes[n] = (uint16_t)(v >> (64 - bits));
    }
    return n;
}

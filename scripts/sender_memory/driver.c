/* One driver for every measured sender, selected with -DMETHOD_*. Outside the measured call the
 * program is identical, so (sections of this program) - (sections of METHOD_EMPTY) isolates the
 * sender: its code, its read-only data, the tag key and the keyed BLAKE2b tag that every method
 * sends. The message comes from argv[1] so nothing is constant-folded. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef METHOD_EMPTY
static const uint8_t KEY[] = "chave-de-teste-nao-usar-em-producao";
uint32_t tag32(const uint8_t *msg, uint8_t len, const uint8_t *key, uint8_t keylen);
#endif
#if defined(METHOD_SCHEME)
uint32_t scheme_encode(const uint8_t *msg, uint8_t len, uint16_t *hashes);
#elif defined(METHOD_PPM)
uint32_t ppm_encode(const uint8_t *msg, uint8_t len, uint8_t *out, uint32_t cap_bytes);
#elif defined(METHOD_SMAZ2)
unsigned long smaz2_compress(unsigned char *dst, unsigned long dstlen, unsigned char *s, unsigned long len);
#elif defined(METHOD_UNISHOX2)
int unishox2_compress_simple(const char *in, int len, char *out);
#endif

int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "";
    uint8_t len = (uint8_t)strlen(in);
    static uint8_t out[1024];
    static uint16_t hashes[128];
    uint32_t t = 0, r = 0;
    /* the I/O buffers belong to the caller: written in every variant so they cancel in the delta */
    out[len] = len;
    hashes[len & 127] = len;
#ifndef METHOD_EMPTY
    t = tag32((const uint8_t *)in, len, KEY, (uint8_t)(sizeof KEY - 1));
#endif
#if defined(METHOD_SCHEME)
    r = scheme_encode((const uint8_t *)in, len, hashes);
#elif defined(METHOD_PPM)
    r = ppm_encode((const uint8_t *)in, len, out, sizeof out);
#elif defined(METHOD_SMAZ2)
    r = (uint32_t)smaz2_compress(out, sizeof out, (unsigned char *)in, len);
#elif defined(METHOD_UNISHOX2)
    r = (uint32_t)unishox2_compress_simple(in, len, (char *)out);
#endif
    printf("%u %u %u %u %u\n", t, r, out[0], hashes[0], len);
    return 0;
}

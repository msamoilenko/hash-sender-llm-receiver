/* Test harness: one hex message per stdin line -> "<nbits> <roundtrip 0|1>" per stdout line.
 * Built with -DFLIP_FIRST_BIT only by the test that proves the roundtrip flag can fail. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

uint32_t ppm_encode(const uint8_t *msg, uint8_t len, uint8_t *out, uint32_t cap_bytes);
int ppm_decode(const uint8_t *in, uint32_t nbits, uint8_t len, uint8_t *msg);

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

int main(void) {
    static char line[1024];
    static uint8_t msg[256], back[256], code[4096];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strcspn(line, "\r\n");
        if (n % 2 || n / 2 > 255) return 2;
        uint8_t len = (uint8_t)(n / 2);
        for (size_t i = 0; i < len; i++) {
            int hi = hexval(line[2 * i]), lo = hexval(line[2 * i + 1]);
            if (hi < 0 || lo < 0) return 2;
            msg[i] = (uint8_t)(hi * 16 + lo);
        }
        uint32_t nbits = ppm_encode(msg, len, code, sizeof code);
        if (nbits > 8 * sizeof code) return 3;
#ifdef FLIP_FIRST_BIT
        code[0] ^= 0x80;
#endif
        memset(back, 0, sizeof back);
        ppm_decode(code, nbits, len, back);
        printf("%u %d\n", nbits, memcmp(back, msg, len) == 0);
    }
    return 0;
}

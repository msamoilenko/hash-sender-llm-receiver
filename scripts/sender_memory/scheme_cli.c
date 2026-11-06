/* Test harness: one hex message per stdin line -> "h0 h1 ... | tag" (decimal) per stdout line. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

uint32_t scheme_encode(const uint8_t *msg, uint8_t len, uint16_t *hashes);
uint32_t tag32(const uint8_t *msg, uint8_t len, const uint8_t *key, uint8_t keylen);

static const char TAG_KEY[] = "chave-de-teste-nao-usar-em-producao";

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

int main(void) {
    static char line[1024];
    static uint8_t msg[256];
    static uint16_t hashes[128];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strcspn(line, "\r\n");
        if (n % 2 || n / 2 > 255) return 2;
        uint8_t len = (uint8_t)(n / 2);
        for (size_t i = 0; i < len; i++) {
            int hi = hexval(line[2 * i]), lo = hexval(line[2 * i + 1]);
            if (hi < 0 || lo < 0) return 2;
            msg[i] = (uint8_t)(hi * 16 + lo);
        }
        uint32_t k = scheme_encode(msg, len, hashes);
        for (uint32_t i = 0; i < k; i++) printf("%u ", hashes[i]);
        printf("| %u\n", tag32(msg, len, (const uint8_t *)TAG_KEY, (uint8_t)strlen(TAG_KEY)));
    }
    return 0;
}

/* PPM-C sender (measurement instrument, not a new scheme). Tables come from a generated header
 * passed with `-include`. Returns the number of code bits; bits beyond cap_bytes are counted but
 * not written. The message length travels in the 8-bit size field, so no end symbol is coded. */
#include "ppm_model.h"

typedef struct {
    ppm_interval_t iv;
    uint32_t follow, nbits, cap_bits;
    uint8_t *out;
} ppm_enc_t;

static void put_bit(ppm_enc_t *e, int b) {
    if (e->nbits < e->cap_bits && b) e->out[e->nbits >> 3] |= (uint8_t)(0x80u >> (e->nbits & 7));
    e->nbits++;
}

static void bit_follow(ppm_enc_t *e, int b) {
    put_bit(e, b);
    for (; e->follow; e->follow--) put_bit(e, !b);
}

static void enc_range(ppm_enc_t *e, uint32_t cum_lo, uint32_t cum_hi, uint32_t total) {
    ppm_narrow(&e->iv, cum_lo, cum_hi, total);
    for (;;) {
        if (e->iv.high < HALF) {
            bit_follow(e, 0);
        } else if (e->iv.low >= HALF) {
            bit_follow(e, 1);
            e->iv.low -= HALF;
            e->iv.high -= HALF;
        } else if (e->iv.low >= Q1 && e->iv.high < Q3) {
            e->follow++;
            e->iv.low -= Q1;
            e->iv.high -= Q1;
        } else {
            break;
        }
        e->iv.low <<= 1;
        e->iv.high = (e->iv.high << 1) | 1;
    }
}

static void enc_symbol(ppm_enc_t *e, const uint8_t *msg, uint32_t i, uint8_t s) {
    uint8_t ex[32];
    memset(ex, 0, sizeof ex);
    uint32_t nex = 0;
    uint32_t top = i < PPM_ORDER ? i : PPM_ORDER;
    for (int32_t k = (int32_t)top; k >= 0; k--) {
        int32_t ci = ppm_find((uint32_t)k, msg, i);
        if (ci < 0) continue;
        const ppm_order_t *o = &PPM_ORDERS[k];
        uint32_t a = ppm_rd(o->off, o->offw, (uint32_t)ci), b = ppm_rd(o->off, o->offw, (uint32_t)ci + 1);
        uint32_t n = 0, t = 0, below = 0, hit = 0;
        for (uint32_t j = a; j < b; j++) {
            uint8_t sym = o->sym[j];
            if (ppm_is_ex(ex, sym)) continue;
            uint32_t c = ppm_rd(o->cnt, o->cntw, j);
            if (sym < s) below += c;
            if (sym == s) hit = c;
            n += c;
            t++;
        }
        if (t == 0) continue;
        if (hit) {
            enc_range(e, below, below + hit, n + t);
            return;
        }
        enc_range(e, n, n + t, n + t);
        for (uint32_t j = a; j < b; j++) {
            if (!ppm_is_ex(ex, o->sym[j])) {
                ppm_set_ex(ex, o->sym[j]);
                nex++;
            }
        }
    }
    uint32_t idx = 0;
    for (uint32_t v = 0; v < s; v++) idx += !ppm_is_ex(ex, (uint8_t)v);
    enc_range(e, idx, idx + 1, 256 - nex);
}

uint32_t ppm_encode(const uint8_t *msg, uint8_t len, uint8_t *out, uint32_t cap_bytes) {
    ppm_enc_t e = {{0, TOP_VALUE}, 0, 0, cap_bytes * 8, out};
    memset(out, 0, cap_bytes);
    for (uint32_t i = 1; i <= len; i++) enc_symbol(&e, msg, i, msg[i - 1]);
    e.follow++;
    bit_follow(&e, e.iv.low >= Q1);
    return e.nbits;
}

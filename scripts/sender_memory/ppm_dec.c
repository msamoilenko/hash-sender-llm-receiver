/* Test-only PPM-C decoder: proves the encoder output is decodable. Not part of the sender size. */
#include "ppm_model.h"

typedef struct {
    ppm_interval_t iv;
    uint64_t value;
    const uint8_t *in;
    uint32_t nbits, pos;
} ppm_dec_t;

static int next_bit(ppm_dec_t *d) {
    int b = d->pos < d->nbits ? (d->in[d->pos >> 3] >> (7 - (d->pos & 7))) & 1 : 0;
    d->pos++;
    return b;
}

static uint32_t dec_target(ppm_dec_t *d, uint32_t total) {
    uint64_t range = d->iv.high - d->iv.low + 1;
    return (uint32_t)(((d->value - d->iv.low + 1) * total - 1) / range);
}

static void dec_range(ppm_dec_t *d, uint32_t cum_lo, uint32_t cum_hi, uint32_t total) {
    ppm_narrow(&d->iv, cum_lo, cum_hi, total);
    for (;;) {
        if (d->iv.high < HALF) {
        } else if (d->iv.low >= HALF) {
            d->value -= HALF;
            d->iv.low -= HALF;
            d->iv.high -= HALF;
        } else if (d->iv.low >= Q1 && d->iv.high < Q3) {
            d->value -= Q1;
            d->iv.low -= Q1;
            d->iv.high -= Q1;
        } else {
            break;
        }
        d->iv.low <<= 1;
        d->iv.high = (d->iv.high << 1) | 1;
        d->value = (d->value << 1) | (uint64_t)next_bit(d);
    }
}

static uint8_t dec_symbol(ppm_dec_t *d, const uint8_t *msg, uint32_t i) {
    uint8_t ex[32];
    memset(ex, 0, sizeof ex);
    uint32_t nex = 0;
    uint32_t top = i < PPM_ORDER ? i : PPM_ORDER;
    for (int32_t k = (int32_t)top; k >= 0; k--) {
        int32_t ci = ppm_find((uint32_t)k, msg, i);
        if (ci < 0) continue;
        const ppm_order_t *o = &PPM_ORDERS[k];
        uint32_t a = ppm_rd(o->off, o->offw, (uint32_t)ci), b = ppm_rd(o->off, o->offw, (uint32_t)ci + 1);
        uint32_t n = 0, t = 0;
        for (uint32_t j = a; j < b; j++) {
            if (ppm_is_ex(ex, o->sym[j])) continue;
            n += ppm_rd(o->cnt, o->cntw, j);
            t++;
        }
        if (t == 0) continue;
        uint32_t target = dec_target(d, n + t);
        if (target < n) {
            /* entries are sorted by symbol, so this walk is the encoder's cumulative order */
            uint32_t cum = 0;
            for (uint32_t j = a; j < b; j++) {
                if (ppm_is_ex(ex, o->sym[j])) continue;
                uint32_t c = ppm_rd(o->cnt, o->cntw, j);
                if (target < cum + c) {
                    dec_range(d, cum, cum + c, n + t);
                    return o->sym[j];
                }
                cum += c;
            }
        }
        dec_range(d, n, n + t, n + t);
        for (uint32_t j = a; j < b; j++) {
            if (!ppm_is_ex(ex, o->sym[j])) {
                ppm_set_ex(ex, o->sym[j]);
                nex++;
            }
        }
    }
    uint32_t target = dec_target(d, 256 - nex), idx = 0;
    for (uint32_t v = 0; v < 256; v++) {
        if (ppm_is_ex(ex, (uint8_t)v)) continue;
        if (idx == target) {
            dec_range(d, idx, idx + 1, 256 - nex);
            return (uint8_t)v;
        }
        idx++;
    }
    return 0;
}

int ppm_decode(const uint8_t *in, uint32_t nbits, uint8_t len, uint8_t *msg) {
    ppm_dec_t d = {{0, TOP_VALUE}, 0, in, nbits, 0};
    for (int j = 0; j < CODE_BITS; j++) d.value = (d.value << 1) | (uint64_t)next_bit(&d);
    for (uint32_t i = 1; i <= len; i++) msg[i - 1] = dec_symbol(&d, msg, i);
    return 0;
}

/* PPM-C with exclusion over the static tables of ppm_tables.py, shared by the encoder (measured)
 * and the test-only decoder. Mirrors baseline_classical.ppm_prob: contexts from the longest order
 * down to 0, a context absent or with no non-excluded symbol is skipped without coding anything,
 * escape frequency = number of non-excluded symbols, then uniform over the non-excluded bytes.
 * Arithmetic coder: Witten-Neal-Cleary (CACM 1987), bit oriented, 40-bit code values. */
#ifndef PPM_MODEL_H
#define PPM_MODEL_H
#include <stdint.h>
#include <string.h>

#define CODE_BITS 40
#define TOP_VALUE ((1ULL << CODE_BITS) - 1)
#define HALF (1ULL << (CODE_BITS - 1))
#define Q1 (HALF >> 1)
#define Q3 (Q1 * 3)
#define MAX_TOTAL (1UL << 23) /* range * total must fit in 64 bits */

typedef struct {
    const uint8_t *ctx;
    uint32_t nctx;
    const void *off;
    uint8_t offw;
    const uint8_t *sym;
    const void *cnt;
    uint8_t cntw;
} ppm_order_t;

#define PPM_ORD(k, ctxp) \
    {ctxp, PPM_NCTX##k, ppm_off##k, sizeof(ppm_off##k##_t), ppm_sym##k, ppm_cnt##k, sizeof(ppm_cnt##k##_t)}

static const ppm_order_t PPM_ORDERS[PPM_ORDER + 1] = {
    PPM_ORD(0, 0),
#if PPM_ORDER >= 1
    PPM_ORD(1, ppm_ctx1),
#endif
#if PPM_ORDER >= 2
    PPM_ORD(2, ppm_ctx2),
#endif
#if PPM_ORDER >= 3
    PPM_ORD(3, ppm_ctx3),
#endif
};

static uint32_t ppm_rd(const void *p, uint8_t w, uint32_t i) {
    if (w == 1) return ((const uint8_t *)p)[i];
    if (w == 2) return ((const uint16_t *)p)[i];
    return ((const uint32_t *)p)[i];
}

/* byte j of "\n" + msg */
static uint8_t ppm_seq(const uint8_t *msg, uint32_t j) { return j ? msg[j - 1] : (uint8_t)'\n'; }

/* index of the context of length k ending before seq position i, or -1 */
static int32_t ppm_find(uint32_t k, const uint8_t *msg, uint32_t i) {
    const ppm_order_t *o = &PPM_ORDERS[k];
    if (k == 0) return o->nctx ? 0 : -1;
    uint8_t c[PPM_ORDER];
    for (uint32_t j = 0; j < k; j++) c[j] = ppm_seq(msg, i - k + j);
    int32_t lo = 0, hi = (int32_t)o->nctx - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) / 2;
        int r = memcmp(o->ctx + (uint32_t)mid * k, c, k);
        if (r == 0) return mid;
        if (r < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static int ppm_is_ex(const uint8_t *ex, uint8_t s) { return (ex[s >> 3] >> (s & 7)) & 1; }
static void ppm_set_ex(uint8_t *ex, uint8_t s) { ex[s >> 3] |= (uint8_t)(1u << (s & 7)); }

typedef struct {
    uint64_t low, high;
} ppm_interval_t;

/* narrow [low, high] to [cum_lo, cum_hi) of total */
static void ppm_narrow(ppm_interval_t *iv, uint32_t cum_lo, uint32_t cum_hi, uint32_t total) {
    uint64_t range = iv->high - iv->low + 1;
    iv->high = iv->low + range * cum_hi / total - 1;
    iv->low = iv->low + range * cum_lo / total;
}
#endif

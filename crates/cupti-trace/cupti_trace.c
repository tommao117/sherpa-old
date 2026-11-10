// Sherpa CUPTI injection tracer.
//
// Usage:
//   CUDA_INJECTION64_PATH=libsherpa_cupti.so SHERPA_TRACE_OUT=/path/trace.txt <program>
//   kill -USR1 <pid>   # start recording kernel / memcpy activity
//   kill -USR2 <pid>   # stop, flush and write SHERPA_TRACE_OUT (overwrites)
//
// Output (text, one record per line):
//   C <cupti_ts_ns> <realtime_ns>             clock calibration at start
//   K <name_id> <start> <end> <graph_id> <stream_id>
//   M <copy_kind> <bytes> <start> <end>
//   S <bytes> <start> <end>                    memset
//   N <name_id> <name>                         name table (at end)
//   E <dropped_records>

#define _GNU_SOURCE
#include <cupti.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BUF_SIZE (8 * 1024 * 1024)
#define NAME_CAP 65536

typedef struct { uint8_t type; uint32_t a; uint64_t b, start, end; uint32_t c, d; } rec_t;

static rec_t *g_recs;
static size_t g_n, g_cap;
static uint64_t g_dropped;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static char *g_names[NAME_CAP];
static uint64_t g_hashes[NAME_CAP];
static uint32_t g_name_ids[NAME_CAP];
static uint32_t g_next_id;
static char *g_id_to_name[NAME_CAP];

static atomic_int g_req_start, g_req_stop, g_recording;
static uint64_t g_calib_cupti, g_calib_real;

static uint64_t fnv(const char *s) {
  uint64_t h = 1469598103934665603ULL;
  for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
  return h;
}

static uint32_t name_id(const char *s) {
  if (!s) s = "<null>";
  uint64_t h = fnv(s);
  size_t i = h % NAME_CAP;
  for (size_t probe = 0; probe < NAME_CAP; probe++, i = (i + 1) % NAME_CAP) {
    if (!g_names[i]) {
      if (g_next_id >= NAME_CAP) return 0;
      g_names[i] = strdup(s); g_hashes[i] = h; g_name_ids[i] = g_next_id;
      g_id_to_name[g_next_id] = g_names[i];
      return g_next_id++;
    }
    if (g_hashes[i] == h && strcmp(g_names[i], s) == 0) return g_name_ids[i];
  }
  return 0;
}

static void push(rec_t r) {
  if (g_n == g_cap) {
    size_t nc = g_cap ? g_cap * 2 : (1u << 20);
    rec_t *p = realloc(g_recs, nc * sizeof(rec_t));
    if (!p) { g_dropped++; return; }
    g_recs = p; g_cap = nc;
  }
  g_recs[g_n++] = r;
}

static void CUPTIAPI buf_req(uint8_t **buf, size_t *size, size_t *max) {
  *buf = aligned_alloc(8, BUF_SIZE);
  *size = BUF_SIZE;
  *max = 0;
}

static void CUPTIAPI buf_done(CUcontext ctx, uint32_t sid, uint8_t *buf, size_t size, size_t valid) {
  (void)ctx; (void)sid; (void)size;
  CUpti_Activity *r = NULL;
  pthread_mutex_lock(&g_mu);
  while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      CUpti_ActivityKernel10 *k = (CUpti_ActivityKernel10 *)r;
      rec_t x = {'K', name_id(k->name), 0, k->start, k->end, k->graphId, k->streamId};
      push(x);
    } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)r;
      rec_t x = {'M', m->copyKind, m->bytes, m->start, m->end, 0, 0};
      push(x);
    } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
      CUpti_ActivityMemset4 *m = (CUpti_ActivityMemset4 *)r;
      rec_t x = {'S', 0, m->bytes, m->start, m->end, 0, 0};
      push(x);
    }
  }
  size_t d = 0;
  cuptiActivityGetNumDroppedRecords(ctx, sid, &d);
  g_dropped += d;
  pthread_mutex_unlock(&g_mu);
  free(buf);
}

static void set_enabled(int on) {
  CUpti_ActivityKind kinds[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
                                CUPTI_ACTIVITY_KIND_MEMSET};
  for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++) {
    if (on) cuptiActivityEnable(kinds[i]); else cuptiActivityDisable(kinds[i]);
  }
}

static void dump(void) {
  const char *out = getenv("SHERPA_TRACE_OUT");
  if (!out) out = "/tmp/sherpa_trace.txt";
  char tmp[4096];
  snprintf(tmp, sizeof(tmp), "%s.tmp", out);
  FILE *f = fopen(tmp, "w");
  if (!f) return;
  pthread_mutex_lock(&g_mu);
  fprintf(f, "C %lu %lu\n", (unsigned long)g_calib_cupti, (unsigned long)g_calib_real);
  for (size_t i = 0; i < g_n; i++) {
    rec_t *r = &g_recs[i];
    if (r->type == 'K')
      fprintf(f, "K %u %lu %lu %u %u\n", r->a, (unsigned long)r->start, (unsigned long)r->end, r->c, r->d);
    else if (r->type == 'M')
      fprintf(f, "M %u %lu %lu %lu\n", r->a, (unsigned long)r->b, (unsigned long)r->start, (unsigned long)r->end);
    else
      fprintf(f, "S %lu %lu %lu\n", (unsigned long)r->b, (unsigned long)r->start, (unsigned long)r->end);
  }
  for (uint32_t i = 0; i < g_next_id; i++) fprintf(f, "N %u %s\n", i, g_id_to_name[i]);
  fprintf(f, "E %lu\n", (unsigned long)g_dropped);
  g_n = 0; g_dropped = 0;
  pthread_mutex_unlock(&g_mu);
  fclose(f);
  rename(tmp, out);
}

static void *control_loop(void *arg) {
  (void)arg;
  for (;;) {
    if (atomic_exchange(&g_req_start, 0) && !atomic_load(&g_recording)) {
      pthread_mutex_lock(&g_mu); g_n = 0; g_dropped = 0; pthread_mutex_unlock(&g_mu);
      struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
      cuptiGetTimestamp(&g_calib_cupti);
      g_calib_real = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
      set_enabled(1);
      atomic_store(&g_recording, 1);
    }
    if (atomic_exchange(&g_req_stop, 0) && atomic_load(&g_recording)) {
      set_enabled(0);
      cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
      atomic_store(&g_recording, 0);
      dump();
    }
    usleep(10000);
  }
  return NULL;
}

static void on_sig(int s) {
  if (s == SIGUSR1) atomic_store(&g_req_start, 1);
  else if (s == SIGUSR2) atomic_store(&g_req_stop, 1);
}

__attribute__((visibility("default"))) int InitializeInjection(void) {
  cuptiActivityRegisterCallbacks(buf_req, buf_done);
  cuptiActivityFlushPeriod(500);
  signal(SIGUSR1, on_sig);
  signal(SIGUSR2, on_sig);
  pthread_t t;
  pthread_create(&t, NULL, control_loop, NULL);
  pthread_detach(t);
  return 1;
}

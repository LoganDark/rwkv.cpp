#include "rwkv.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#include <string>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <random>
#include <utility>
#include <algorithm>

#define _FILE_OFFSET_BITS 64
// Puts an optional break point, if debug is enabled.
#define RWKV_MAYBE_BREAK

#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#    define stat _stat64
#    define fstat _fstat64
#    define ftell _ftelli64
#    define fseek _fseeki64
#    if !defined(NDEBUG)
#        include <intrin.h>
#        define RWKV_MAYBE_BREAK __debugbreak()
#    endif
#else
#    if !defined(__APPLE__)
#        define ftell ftello
#        define fseek fseeko
#    endif
#endif

static_assert(sizeof(stat::st_size) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");
static_assert(sizeof(decltype(ftell(NULL))) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");

#define RWKV_MAX_NODES 80000

#include "rwkv_error_handling.inc"

#include "rwkv_utilities.inc"

#include "rwkv_file_format.inc"

#include "rwkv_model_loading.inc"

#include "rwkv_operators.inc"

#include "rwkv_graph.inc"

// API function.
struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads, const uint32_t n_gpu_layers) {
    global_last_error = RWKV_ERROR_NONE;

    std::unique_ptr<struct rwkv_context> ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, ctx, "Failed to allocate rwkv_context");

    ctx->model = new(std::nothrow) struct rwkv_model();
    ctx->model->reference_count++;

    ctx->n_threads = n_threads;

    if (n_gpu_layers) {
        ggml_backend_t backend = nullptr;

#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        RWKV_ENSURE_OR_NULL(backend);
#endif

#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
        RWKV_ENSURE_OR_NULL(backend);
#endif

#ifdef GGML_USE_BLAS
        backend = ggml_backend_blas_init();
        RWKV_ENSURE_OR_NULL(backend);
        ggml_backend_blas_set_n_threads(backend, ctx->n_threads);
#endif
        if (backend != nullptr) {
            ctx->model->backends.push_back(backend);
        }
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    RWKV_ENSURE_OR_NULL(cpu_backend);
    ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
    ctx->model->backends.push_back(cpu_backend);

    int ngl = n_gpu_layers;
    if (ctx->model->backends.size() == 1) {
        ngl = 0;
    }

    RWKV_ENSURE_OR_NULL(rwkv_load_model_from_file(file_path, *ctx->model, ngl));

    RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*ctx->model, ctx->serial_graph));

    return ctx.release();
}

// API function.
struct rwkv_context * rwkv_clone_context(struct rwkv_context * ctx, const uint32_t n_threads) {
    std::unique_ptr<struct rwkv_context> clone(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, clone, "Failed to allocate rwkv_context");

    clone->model = ctx->model;
    clone->model->reference_count++;

    clone->n_threads = n_threads;

    RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*clone->model, clone->serial_graph));

    clone->last_used_sequence_length = 0;

    clone->print_errors = ctx->print_errors;

    return clone.release();
}

#include "rwkv_eval.inc"

// API function.
// Provided for backwards compatibility.
extern "C" RWKV_API uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_state_len(ctx);
}

// API function.
// Provided for backwards compatibility.
extern "C" RWKV_API uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_logits_len(ctx);
}

// API function.
size_t rwkv_get_n_vocab(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
}

// API function.
size_t rwkv_get_n_embed(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_embed;
}

// API function.
size_t rwkv_get_n_layer(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_layer;
}

// API function.
size_t rwkv_get_state_len(const struct rwkv_context * ctx) {
    const struct rwkv_file_header & header = ctx->model->header;

    if (ctx->model->arch_version_major >= 5) {
        return (size_t) header.n_embed * (2 + ctx->model->head_size) * (size_t) header.n_layer;
    } else {
        return (size_t) header.n_embed * 5 * (size_t) header.n_layer;
    }
}

// API function.
size_t rwkv_get_logits_len(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
}

// API function.
void rwkv_free(struct rwkv_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (--ctx->model->reference_count == 0) {
        for (auto buffer : ctx->model->buffers_w) {
            ggml_backend_buffer_free(buffer);
        }

        for (auto backend : ctx->model->backends) {
            ggml_backend_free(backend);
        }

        ggml_free(ctx->model->ggml_ctx);

        delete ctx->model;
    }

    ggml_backend_sched_free(ctx->serial_graph.sched);
    ggml_free(ctx->serial_graph.ggml_ctx);

    if (ctx->last_used_sequence_length > 0) {
        ggml_backend_sched_free(ctx->sequential_graph.sched);
        ggml_free(ctx->sequential_graph.ggml_ctx);
    }

    delete ctx;
}

// API function.
void rwkv_set_print_errors(struct rwkv_context * ctx, const bool print_errors) {
    bool * ptr = ctx ? &ctx->print_errors : &global_print_errors;
    *ptr = print_errors;
}

// API function.
bool rwkv_get_print_errors(const struct rwkv_context * ctx) {
    return ctx ? ctx->print_errors : global_print_errors;
}

// API function.
enum rwkv_error_flags rwkv_get_last_error(struct rwkv_context * ctx) {
    enum rwkv_error_flags * ptr = ctx ? &ctx->last_error : &global_last_error;
    enum rwkv_error_flags value = *ptr;
    *ptr = RWKV_ERROR_NONE;
    return value;
}

#include "rwkv_quantize.inc"

// API function.
const char * rwkv_get_system_info_string(void) {
    static std::string s;

    if (s.empty()) {
        s  = "";
        s += "AVX="       + std::to_string(ggml_cpu_has_avx())       + " ";
        s += "AVX2="      + std::to_string(ggml_cpu_has_avx2())      + " ";
        s += "AVX512="    + std::to_string(ggml_cpu_has_avx512())    + " ";
        s += "FMA="       + std::to_string(ggml_cpu_has_fma())       + " ";
        s += "NEON="      + std::to_string(ggml_cpu_has_neon())      + " ";
        s += "ARM_FMA="   + std::to_string(ggml_cpu_has_arm_fma())   + " ";
        s += "F16C="      + std::to_string(ggml_cpu_has_f16c())      + " ";
        s += "FP16_VA="   + std::to_string(ggml_cpu_has_fp16_va())   + " ";
        s += "WASM_SIMD=" + std::to_string(ggml_cpu_has_wasm_simd()) + " ";
        s += "SSE3="      + std::to_string(ggml_cpu_has_sse3())      + " ";
        s += "VSX="       + std::to_string(ggml_cpu_has_vsx());
    }

    return s.c_str();
}

#include "rwkv_vocab_v20230424.h"
#include "rwkv_vocab_v20230424_accel.h"

size_t rwkv_vocab_v20230424_encode(const char * data, const size_t len, uint32_t * out, const size_t out_len) {
    size_t count = 0;
    uint32_t last_token = 0;

    for (size_t start = 0; start < len;) {
        const struct rwkv_vocab_v20230424_accel_entry * last = &rwkv_vocab_v20230424_accel;

        for (size_t i = start; i < len; i++) {
            const uint8_t byte = ((const uint8_t *) data)[i];
            bool found2 = false;

            for (size_t c = 0; c < last->num_children; c++) {
                if (last->children[c].byte == byte) {
                    found2 = true;
                    last = &last->children[c];

                    if (last->token > 0) {
                        last_token = last->token;
                        start = i + 1;
                    }

                    break;
                }
            }

            if (!found2) {
                break;
            }
        }

        if (last_token) {
            if (out) {
                if (count < out_len) {
                    out[count] = last_token;
                } else {
                    return count;
                }
            }

            count++;
            last_token = 0;
        } else {
            break;
        }
    }

    return count;
}

size_t rwkv_vocab_v20230424_decode(const uint32_t * tokens, const size_t len, char * out, const size_t out_len) {
    size_t count = 0;

    for (size_t i = 0; i < len; i++) {
        const uint32_t token = tokens[i];

        if (token < sizeof(rwkv_vocab_v20230424) / sizeof(rwkv_vocab_v20230424_entry)) {
            const struct rwkv_vocab_v20230424_entry * entry = &rwkv_vocab_v20230424[token];

            if (out) {
                if (count + entry->len <= out_len) {
                    memcpy(&out[count], entry->bytes, entry->len);
                } else {
                    memcpy(&out[count], entry->bytes, out_len - count);
                    return out_len;
                }
            }

            count += entry->len;
        }
    }

    return count;
}

// https://codereview.stackexchange.com/a/180527
void rwkv_softmax(const float * logits, const size_t n_vocab, float * output) {
    float max = -INFINITY;
    for (size_t i = 0; i < n_vocab; i++) {
        max = std::max(max, logits[i]);
    }

    float sum = 0.0;
    for (size_t i = 0; i < n_vocab; i++) {
        sum += expf(logits[i] - max);
    }

    float offset = max + logf(sum);
    for (size_t i = 0; i < n_vocab; i++) {
        output[i] = expf(logits[i] - offset);
    }
}

void rwkv_temper(const float * probs, const size_t n_vocab, const float temperature, float * out) {
    if (temperature == 1.0) {
        float sum = 0.0;
        for (size_t i = 0; i < n_vocab; i++) {
            sum += probs[i];
        }

        for (size_t i = 0; i < n_vocab; i++) {
            out[i] /= sum;
        }
    } else if (temperature > 0) {
        float sum = 0.0;
        for (size_t i = 0; i < n_vocab; i++) {
            sum += out[i] = powf(probs[i], 1.0 / temperature);
        }

        for (size_t i = 0; i < n_vocab; i++) {
            out[i] /= sum;
        }
    } else {
        float prob_max = 0.0;
        size_t choice = 0;

        for (size_t i = 0; i < n_vocab; i++) {
            const float prob = probs[i];
            if (prob > prob_max) {
                out[choice] = 0.0;
                out[i] = 1.0;
                prob_max = prob;
                choice = i;
            } else {
                out[i] = 0.0;
            }
        }
    }
}

uint32_t rwkv_sample(const float * probs, const size_t n_vocab, const size_t top_k, const float top_p, uint32_t * top) {
    std::unique_ptr<uint32_t []> _(!top && top_k > 1 ? top = new(std::nothrow) uint32_t [n_vocab] : NULL);

    if (!top && top_k > 1) {
        return 0;
    }

    if (top) {
        for (uint32_t token = 0; token < n_vocab; token++) {
            top[token] = token;
        }

        std::stable_sort(top, top + n_vocab, [probs](const uint32_t a, const uint32_t b) {
            return probs[a] > probs[b];
        });
    }

    if (top_k == 0 || n_vocab == 0) {
        return 0;
    } else if (top_k == 1) {
        if (top) {
            return top[0];
        } else {
            float prob_max = 0.0;
            uint32_t choice = 0;

            for (size_t i = 0; i < n_vocab; i++) {
                const float prob = probs[i];
                if (prob > prob_max) {
                    prob_max = prob;
                    choice = (uint32_t) i;
                }
            }

            return choice;
        }
    }

    float prob_included = 0.0;
    for (size_t i = 0; i < top_k && prob_included < top_p; i++) {
        const uint32_t token = top[i];
        const float prob = probs[token];
        prob_included += prob;
    }

    double pick; {
        std::random_device random;
        std::mt19937 gen(random());
        std::uniform_real_distribution<double> dis(0.0, prob_included);
        pick = dis(gen);
    }

    for (size_t i = 0; i < top_k; i++) {
        const uint32_t token = top[i];
        if ((prob_included -= probs[token]) <= pick) {
            return token;
        }
    }

    return 0;
}

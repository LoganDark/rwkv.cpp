#include "ggml.h"
#include "rwkv.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#ifdef _WIN32
bool QueryPerformanceFrequency(uint64_t* lpFrequency);
bool QueryPerformanceCounter(uint64_t* lpPerformanceCount);

#define time_t uint64_t
#define time_calibrate(freq) do { QueryPerformanceFrequency(&freq); freq = (freq + 999) / 1000; } while (0)
#define time_measure(x) QueryPerformanceCounter(&x)
#define TIME_DIFF(freq, start, end) (double) ((end - start) / freq) / 1000.
#else
#include <time.h>

#define time_t struct timespec
#define time_calibrate(freq) (void) freq
#define time_measure(x) clock_gettime(CLOCK_MONOTONIC, &x)
#define TIME_DIFF(freq, start, end) (double) ((end.tv_nsec - start.tv_nsec) / 1000000 + 999) / 1000
#endif

size_t readline(char * dest, const size_t len) {
	if (!fgets(dest, len, stdin)) {
		return 0;
	}

	// https://stackoverflow.com/a/28462221
	const size_t written = strcspn(dest, "\n");
	dest[written] = 0;
	return written;
}

void feed_model(struct rwkv_context * ctx, size_t * state_count, float * state, float * logits, const char * input, const size_t len) {
	uint32_t token = 0;
	for (size_t done = 0; done < len; done += rwkv_vocab_v20230424_decode(&token, 1, NULL, 0)) {
		rwkv_vocab_v20230424_encode(input + done, len - done, &token, 1);
		rwkv_eval(ctx, token, (*state_count)++ ? state : NULL, state, logits);
	}
}

int main() {
#ifdef _WIN32
	// enable UTF-8
	system("chcp 65001 >nul");
#endif

	time_t freq, start, end;
	time_calibrate(freq);

	fprintf(stderr, "%s\n\n", rwkv_get_system_info_string());

	fprintf(stderr, "Enter (world) model path: ");
	char model_path[128];
	const size_t model_path_len = readline(model_path, 128);

	fprintf(stderr, "\nLoading model ...");
	time_measure(start);
	struct rwkv_context * ctx = rwkv_init_from_file(model_path, 6);
	time_measure(end);
	fprintf(stderr, " %.3fs\n", TIME_DIFF(freq, start, end));

	fprintf(stderr, "GPU offload ...");
	time_measure(start);
	rwkv_gpu_offload_layers(ctx, 40);
	time_measure(end);
	fprintf(stderr, " %.3fs\n", TIME_DIFF(freq, start, end));

	fprintf(stderr, "Initializing state ...");
	time_measure(start);

	size_t state_count = 0;
	float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
	float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));
	const size_t n_vocab = rwkv_get_n_vocab(ctx);
	uint32_t * top = calloc(n_vocab, sizeof(uint32_t));
	char decode_buf[128];

	time_measure(end);
	fprintf(stderr, " %.3fs\n\n", TIME_DIFF(freq, start, end));

	fprintf(stderr, "Enter user display name  : ");
	char user_display_name[128];
	const size_t user_display_name_len = readline(user_display_name, 128);

	fprintf(stderr, "Enter model display name : ");
	char model_display_name[128];
	const size_t model_display_name_len = readline(model_display_name, 128);

	fprintf(stderr, "Enter user prompt  : ");
	char user_prompt[128];
	const size_t user_prompt_len = readline(user_prompt, 128);

	fprintf(stderr, "Enter model prompt : ");
	char model_prompt[128];
	const size_t model_prompt_len = readline(model_prompt, 128);

	bool used_full_reverse_prompt = false;

	for (size_t n = 0; ; n++) {
		fprintf(stderr, "\n%.*s: ", (unsigned) user_display_name_len, user_display_name);

		char user_line[16384];
		size_t user_line_len = readline(user_line, 16384);

		char model_feed[16384];
		const size_t model_feed_len = used_full_reverse_prompt
			? snprintf(model_feed, 16384, "%.*s\n\n%.*s", (unsigned) user_line_len, user_line, (unsigned) model_prompt_len, model_prompt)
			: snprintf(model_feed, 16384, "%.*s%.*s\n\n%.*s", (unsigned) user_prompt_len, user_prompt, (unsigned) user_line_len, user_line, (unsigned) model_prompt_len, model_prompt);

		feed_model(ctx, &state_count, state, logits, model_feed, model_feed_len);
		fprintf(stderr, "\n%.*s:", (unsigned) model_display_name_len, model_display_name);

		used_full_reverse_prompt = n > 3;
		char reverse_prompt[128] = { 0 };
		const size_t reverse_prompt_len = !used_full_reverse_prompt
			? snprintf(reverse_prompt, 128, "\n\n")
			: snprintf(reverse_prompt, 128, "\n\n%.*s", (unsigned) user_prompt_len, user_prompt);
		size_t reverse_prompt_progress = 0;

		while (reverse_prompt_progress < reverse_prompt_len) {
			rwkv_softmax(logits, n_vocab, logits);
			//rwkv_temper(logits, n_vocab, 1.0, logits);
			const uint32_t token = rwkv_sample(logits, n_vocab, n_vocab, 0.25, top);

			if (!token) {
				feed_model(ctx, &state_count, state, logits, reverse_prompt + reverse_prompt_progress, reverse_prompt_len - reverse_prompt_progress);
				break;
			}

			char decoded[128];
			const uint32_t decoded_len = rwkv_vocab_v20230424_decode(&token, 1, decoded, 128);

			for (size_t i = 0; i < decoded_len; i++) {
				if (decoded[i] == reverse_prompt[reverse_prompt_progress]) {
					if (++reverse_prompt_progress == reverse_prompt_len) {
						feed_model(ctx, &state_count, state, logits, reverse_prompt, reverse_prompt_len);
						break;
					}
				} else {
					if (reverse_prompt_progress > 0) {
						if (reverse_prompt_progress > i) {
							// The start of the reverse prompt won't have been forwarded to the model, because we will have been waiting for the rest of it.
							feed_model(ctx, &state_count, state, logits, reverse_prompt, reverse_prompt_progress - i);
						}

						fprintf(stderr, "%.*s", (unsigned) reverse_prompt_progress, reverse_prompt);
						reverse_prompt_progress = 0;
					}

					fputc(decoded[i], stderr);
				}
			}

			if (reverse_prompt_progress == 0) {
				rwkv_eval(ctx, token, state, state, logits);
			}
		}

		fprintf(stderr, "\n");
	}

	free(top);
	free(logits);
	free(state);
	rwkv_free(ctx);
}
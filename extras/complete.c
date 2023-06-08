#include "ggml.h"
#include "rwkv.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

int main() {
#ifdef _WIN32
	// enable UTF-8
	system("chcp 65001 >nul");
#endif

	struct rwkv_context * ctx = rwkv_init_from_file("C:\\Users\\LoganDark\\Documents\\RWKV\\RWKV-4-World-1.5B-v1-20230607-ctx4096-Q8_0.bin", 6);
	rwkv_gpu_offload_layers(ctx, 40);

	const char prompt[] = "The common raven is a large all-black passerine bird. It is the most widely distributed of all corvids, found across the Northern Hemisphere.";
	const size_t prompt_len = sizeof(prompt) - 1;

	const size_t max_tokens = 256;
	uint32_t * tokens = calloc(max_tokens, sizeof(uint32_t));
	const size_t prompt_tokens = rwkv_vocab_v20230424_encode(prompt, prompt_len, tokens, max_tokens);

	float * state = calloc(rwkv_get_state_buffer_element_count(ctx), sizeof(float));
	const size_t n_vocab = rwkv_get_logits_buffer_element_count(ctx);
	float * logits = calloc(n_vocab, sizeof(float));

	char decode_buf[128];
	for (size_t i = 0; i < max_tokens; i++) {
		uint32_t token;

		if (i < prompt_tokens) {
			token = tokens[i];
		} else if (token = rwkv_sample(logits, n_vocab, n_vocab, 0.75, 1.0)) {
			tokens[i] = token;
		} else {
			break;
		}

		const size_t decode_len = rwkv_vocab_v20230424_decode(&token, 1, decode_buf, 128);
		printf("%.*s", (unsigned) decode_len, decode_buf);
		rwkv_eval(ctx, token, i == 0 ? NULL : state, state, logits);
		rwkv_softmax(logits, n_vocab, logits);
	}

	free(logits);
	free(state);
	free(tokens);
	rwkv_free(ctx);
}
#include "ggml.h"
#include "rwkv.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

int main() {
	const char task[] = "Hello, World!";
	const size_t task_len = sizeof(task) / sizeof(char) - 1;

	const size_t num_tokens = rwkv_vocab_v20230424_encode(task, task_len, NULL, 0);
	uint32_t * tokens = calloc(num_tokens + 1, sizeof(uint32_t));
	const size_t written = rwkv_vocab_v20230424_encode(task, task_len, tokens, num_tokens + 1);

	printf("Expect: %zu\nTokens:", num_tokens);

	for (size_t i = 0; i < written; i++) {
		printf(" %" PRId32, tokens[i]);
	}

	printf("\nActual: %zu\n", written);

	const size_t len_decoded = rwkv_vocab_v20230424_decode(tokens, written, NULL, 0);
	char * decoded = calloc(len_decoded + 1, sizeof(char));
	const size_t written2 = rwkv_vocab_v20230424_decode(tokens, written, decoded, len_decoded + 1);

	printf("\nExpect: %zu\nDecode: %s\nActual: %zu\n", len_decoded, decoded, written2);

	return EXIT_SUCCESS;
}
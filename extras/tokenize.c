#include "ggml.h"
#include "rwkv.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

int main() {
	const char input[] = "Hello, World!";
	const size_t input_len = sizeof(input) / sizeof(char) - 1;

	printf(" Input: %.*s\n", (unsigned) input_len, input);
	printf("Length: %zu bytes\n", input_len);
	printf(" Bytes:");

	for (size_t i = 0; i < input_len; i++) {
		printf(" %" PRId8, input[i]);
	}

	puts("");

	const size_t num_tokens = rwkv_vocab_v20230424_encode(input, input_len, NULL, 0);
	uint32_t * tokens = calloc(num_tokens + 1, sizeof(uint32_t));
	const size_t written = rwkv_vocab_v20230424_encode(input, input_len, tokens, num_tokens + 1);

	puts("\n------  Encode");
	printf("Expect: %zu tokens\n", num_tokens);
	printf("   Got: %zu tokens\n", written);
	printf("Tokens:");

	for (size_t i = 0; i < written; i++) {
		printf(" %" PRId32, tokens[i]);
	}

	puts("");

	const size_t len_decoded = rwkv_vocab_v20230424_decode(tokens, written, NULL, 0);
	char * decoded = calloc(len_decoded + 1, sizeof(char));
	const size_t written2 = rwkv_vocab_v20230424_decode(tokens, written, decoded, len_decoded + 1);

	puts("\n------  Decode");
	printf("Expect: %zu bytes\n", len_decoded);
	printf("   Got: %zu bytes\n", written2);
	printf("Decode: %.*s\n", (unsigned) written2, decoded);
	printf(" Bytes:");

	for (size_t i = 0; i < written2; i++) {
		printf(" %" PRId8, (uint8_t) decoded[i]);
	}

	puts("");

	return EXIT_SUCCESS;
}
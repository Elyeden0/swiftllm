/**
 * SwiftLLM C API
 *
 * Provides a C-compatible interface to the SwiftLLM universal LLM gateway.
 * This header can be used from C, C++, Go (cgo), Java (Panama FFI / JNI),
 * Ruby (FFI), PHP (FFI), C# (P/Invoke), Elixir (Rustler/NIF), Swift,
 * Kotlin (JNA), Zig, and any other language with C FFI support.
 *
 * Link against: -lswiftllm (the compiled cdylib)
 *
 * Thread safety: Each SwiftLLMHandle is independent and may be used from
 * a single thread at a time. Create separate handles for concurrent use.
 */

#ifndef SWIFTLLM_H
#define SWIFTLLM_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a SwiftLLM instance.
 */
typedef struct SwiftLLMHandle SwiftLLMHandle;

/**
 * Create a new SwiftLLM instance.
 * Returns NULL on failure.
 * Must be freed with swiftllm_destroy().
 */
SwiftLLMHandle *swiftllm_create(void);

/**
 * Destroy a SwiftLLM instance and free all resources.
 * Safe to call with NULL.
 */
void swiftllm_destroy(SwiftLLMHandle *handle);

/**
 * Register a provider.
 *
 * @param handle   A valid SwiftLLM handle.
 * @param name     Provider name: "openai", "anthropic", "gemini", "mistral",
 *                 "ollama", "groq", "together", "bedrock".
 * @param api_key  API key string, or NULL for providers that don't need one.
 * @param base_url Custom base URL, or NULL to use the provider's default.
 * @return 0 on success, -1 on error.
 */
int swiftllm_add_provider(SwiftLLMHandle *handle,
                           const char *name,
                           const char *api_key,
                           const char *base_url);

/**
 * Send a chat completion request.
 *
 * @param handle  A valid SwiftLLM handle with at least one provider configured.
 * @param model   Model identifier (e.g. "gpt-4o-mini", "claude-sonnet-4-6").
 * @param prompt  User message text.
 * @return Heap-allocated JSON string with the OpenAI-compatible response,
 *         or NULL on error. Caller MUST free with swiftllm_free_string().
 */
char *swiftllm_completion(SwiftLLMHandle *handle,
                           const char *model,
                           const char *prompt);

/**
 * One-shot completion — infers the provider from the model name.
 *
 * @param model   Model identifier.
 * @param prompt  User message text.
 * @param api_key API key for the inferred provider.
 * @return Heap-allocated JSON string, or NULL on error.
 *         Caller MUST free with swiftllm_free_string().
 */
char *swiftllm_quick_completion(const char *model,
                                 const char *prompt,
                                 const char *api_key);

/**
 * Free a string returned by any swiftllm function.
 * Safe to call with NULL.
 */
void swiftllm_free_string(char *ptr);

/**
 * Get the library version string.
 * Returns a static string — do NOT free this pointer.
 */
const char *swiftllm_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SWIFTLLM_H */

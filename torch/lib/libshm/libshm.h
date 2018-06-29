#ifndef LIBSHM_H
#define LIBSHM_H

#include <TH/TH.h>

#ifdef __cplusplus
#define EXPORT_API extern "C"
#else
#define EXPORT_API
#endif

typedef struct {
  char *manager_handle;
  // NB: th_context is a temporary field
  THMapAllocatorContext *th_context;
  at::BoundDeleter th_deleter;
} libshm_context;

EXPORT_API void libshm_init(const char *manager_exec_path);
EXPORT_API libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags);
EXPORT_API std::unique_ptr<void, at::BoundDeleter> libshm_alloc(void *_ctx, ptrdiff_t size);
EXPORT_API void libshm_context_free(libshm_context *context);

EXPORT_API at::Deleter* getTHManagedSharedDeleter();

#endif

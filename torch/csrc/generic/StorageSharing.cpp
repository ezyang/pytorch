#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <random>

static PyObject * THPStorage_(sharedDecref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
#ifndef THC_GENERIC_FILE
  libshm_context *ctx = NULL;
  THWStorage *storage = self->cdata;
  if (storage->allocatorVoidPtr == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  }
  if (ctx)
    THRefcountedMapAllocator_decref(ctx->th_context, THWStorage_(data)(storage));
#endif
  Py_INCREF(self);
  return (PyObject *)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(sharedIncref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
#ifndef THC_GENERIC_FILE
  libshm_context *ctx = NULL;
  THWStorage *storage = self->cdata;
  if (storage->allocatorVoidPtr == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  }
  if (ctx)
    THRefcountedMapAllocator_incref(ctx->th_context, THWStorage_(data)(storage));
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newTHView)(THWStorage *base, ptrdiff_t offset, size_t size)
{
  void *data = (char*)base->data<real>() + offset;
  THWStoragePtr view(THWStorage_(newWithData)(LIBRARY_STATE (real*)data, size));
  view->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
  view->view = base;
  THWStorage_(retain)(LIBRARY_STATE base);
  return THPStorage_(New)(view.release());
}

#ifndef THC_GENERIC_FILE
// TODO: move this somewhere - we only need one version
static std::string THPStorage_(__newHandle)() {
  std::random_device rd;
  std::string handle = "/torch_";
#ifdef _MSC_VER
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(getpid());
#endif
  handle += "_";
  handle += std::to_string(rd());
  return handle;
}

static THWStorage* THPStorage_(newFilenameStorage)(ptrdiff_t size)
{
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_EXCLUSIVE;
  std::string handle = THPStorage_(__newHandle)();
  auto ctx = libshm_context_new(NULL, handle.c_str(), flags);
  return THWStorage_(newWithAllocator)(size, &THManagedSharedAllocator, (void*)ctx);
}

static PyObject * THPStorage_(pyNewFilenameStorage)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return NULL;
  }
  return THPStorage_(New)(THPStorage_(newFilenameStorage)(size));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(shareFilename)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  libshm_context *ctx;
  // Storage is already in shared memory, just return a handle
  if (storage->allocatorVoidPtr == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    THWStoragePtr new_storage(THPStorage_(newFilenameStorage)(storage->size));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = (libshm_context*)storage->allocatorContext;
  }

  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle));
  if (!manager_handle) return NULL;
  THPObjectPtr storage_handle(
    PyBytes_FromString(THMapAllocatorContext_filename(ctx->th_context)));
  if (!storage_handle) return NULL;
  THPObjectPtr size(PyLong_FromLong(storage->size));
  if (!size) return NULL;

  THPObjectPtr tuple(PyTuple_New(3));
  if (!tuple) return NULL;
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedFilename)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  PyObject *_manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject *_object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size = PyTuple_GET_ITEM(args, 2);
  if (!PyBytes_Check(_manager_handle) || !PyBytes_Check(_object_handle) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, NULL, "_new_shared in file system mode", 1,
        "a handle (string/bytes) and storage size (int)");
    return NULL;
  }
  const char *manager_handle = PyBytes_AS_STRING(_manager_handle);
  const char *object_handle = PyBytes_AS_STRING(_object_handle);
  int64_t size = THPUtils_unpackLong(_size);
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_NOCREATE;
  libshm_context *ctx = libshm_context_new(manager_handle, object_handle, flags);
  return THPStorage_(New)(THWStorage_(newWithAllocator)(size,
      &THManagedSharedAllocator, (void*)ctx));
  END_HANDLE_TH_ERRORS
}

static THWStorage* THPStorage_(newFdStorage)(ptrdiff_t size)
{
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_EXCLUSIVE |
              TH_ALLOCATOR_MAPPED_KEEPFD |
              TH_ALLOCATOR_MAPPED_UNLINK;
  std::string handle = THPStorage_(__newHandle)();
  auto ctx = THMapAllocatorContext_new(handle.c_str(), flags);
  return THWStorage_(newWithAllocator)(size, &THMapAllocator, (void*)ctx);
}

static PyObject * THPStorage_(pyNewFdStorage)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return NULL;
  }
  return THPStorage_(New)(THPStorage_(newFdStorage)(size));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(shareFd)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  THMapAllocatorContext *ctx;
  // Storage is already in shared memory, just return a handle
  if (storage->allocatorVoidPtr == &THMapAllocator) {
    ctx = (THMapAllocatorContext*)storage->allocatorContext;
  } else {
    THWStoragePtr new_storage(THPStorage_(newFdStorage)(storage->size));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = (THMapAllocatorContext*)storage->allocatorContext;
  }

  THPObjectPtr storage_handle(PyLong_FromLong(THMapAllocatorContext_fd(ctx)));
  if (!storage_handle) return NULL;
  THPObjectPtr size(PyLong_FromLong(storage->size));
  if (!size) return NULL;

  THPObjectPtr tuple(PyTuple_New(2));
  if (!tuple) return NULL;
  PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedFd)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject *_tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject *_size = PyTuple_GET_ITEM(args, 1);
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, NULL, "_new_shared in file descriptor mode",
        1, "a file descriptor (int) and storage size (int)");
    return NULL;
  }
  int fd;
  int tmp_fd = (int) THPUtils_unpackLong(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  if ((fd = dup(tmp_fd)) == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return NULL;
  }

  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_NOCREATE |
              TH_ALLOCATOR_MAPPED_KEEPFD |
              TH_ALLOCATOR_MAPPED_FROMFD;
  THMapAllocatorContext *ctx = THMapAllocatorContext_newWithFd(NULL, fd, flags);
  return THPStorage_(New)(THWStorage_(newWithAllocator)(size, &THMapAllocator,
      (void*)ctx));
  END_HANDLE_TH_ERRORS
}

#else // THC_GENERIC_FILE

static PyObject * THPStorage_(shareCuda)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  at::DeviceGuard device_guard(storage->device);
  THPObjectPtr tuple(PyTuple_New(5));
  THPObjectPtr device(PyLong_FromLong(storage->device));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size(PyLong_FromLong(storage->size));
  THPObjectPtr _offset(PyLong_FromLong(0));
  THPObjectPtr view_size(PyLong_FromLong(storage->size));
  if (THWStorage_(data)(LIBRARY_STATE storage)) {
    size_t base_size;
    void *base_ptr = THCCachingAllocator_getBaseAllocation(THWStorage_(data)(LIBRARY_STATE storage), &base_size);
    ptrdiff_t offset = (char*)storage->data<real>() - (char*)base_ptr;

    cudaIpcMemHandle_t handle;
    THCudaCheck(cudaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char *)&handle, CUDA_IPC_HANDLE_SIZE);
    _offset = PyLong_FromSsize_t((Py_ssize_t)offset);
    size = PyLong_FromSize_t(base_size / sizeof(real));
  }
  if (!tuple || !device || !_handle || !size || !_offset || !view_size) {
    return NULL;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  PyTuple_SET_ITEM(tuple.get(), 3, _offset.release());
  PyTuple_SET_ITEM(tuple.get(), 4, view_size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedCuda)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 5, "tuple of 5 items expected");
  PyObject *_device = PyTuple_GET_ITEM(args, 0);
  PyObject *_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size = PyTuple_GET_ITEM(args, 2);
  PyObject *_offset = PyTuple_GET_ITEM(args, 3);
  PyObject *_view_size = PyTuple_GET_ITEM(args, 4);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size)
      && (_handle == Py_None || PyBytes_Check(_handle))
      && THPUtils_checkLong(_offset) && THPUtils_checkLong(_view_size))) {
    THPUtils_invalidArguments(args, NULL, "_new_shared in CUDA mode", 1,
        "(int device, bytes handle, int storage_size, int offset, int view_size");
    return NULL;
  }

  size_t storage_size = (size_t)THPUtils_unpackLong(_size);
  ptrdiff_t offset = (ptrdiff_t)THPUtils_unpackLong(_offset);
  size_t view_size =  (size_t)THPUtils_unpackLong(_view_size);

  int64_t device = THPUtils_unpackLong(_device);
  at::DeviceGuard device_guard(device);

  char *buffer;
  Py_ssize_t handle_size;
  if (PyBytes_AsStringAndSize(_handle, &buffer, &handle_size) == -1) {
    return NULL;
  }
  THPUtils_assert(handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle size");
  cudaIpcMemHandle_t handle = *(cudaIpcMemHandle_t*)buffer;

  void *devPtr = NULL;
  THCudaCheck(cudaIpcOpenMemHandle(&devPtr, handle, cudaIpcMemLazyEnablePeerAccess));

  THWStoragePtr base(THWStorage_(newWithDataAndAllocator)(
      LIBRARY_STATE (real*)devPtr, storage_size, &THCIpcAllocator, (void*)device));
  base->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_FREEMEM;

  if (offset != 0 || view_size != storage_size) {
    return THPStorage_(newTHView)(base.get(), offset, view_size);
  }

  return THPStorage_(New)(base.release());
  END_HANDLE_TH_ERRORS
}
#endif

// Returns an object that holds a "weak" pointer to the THStorage.  This
// pointer keeps the THStorage struct live, but does not retain the data
// pointer.
//
// NB: This does NOT preserve object identity when you call it multiple times
static PyObject * THPStorage_(weakRef)(THPStorage *self, PyObject *weak_ref_class) {
  HANDLE_TH_ERRORS
  THStorage* storage = self->cdata;
  while (storage->flag & TH_STORAGE_VIEW) {
    storage = storage->view;
  }

  THStorage_weakRetain(storage);

  THPObjectPtr args(Py_BuildValue("(N)", PyLong_FromVoidPtr(storage)));
  if (!args) return NULL;
  THPObjectPtr ref(PyObject_Call(weak_ref_class, args, NULL));
  if (!ref) return NULL;

  // We need to also add a finalizer with an owning reference to the weak class,
  // so that we can keep the "weak" object live until it should actually be
  // cleared form the map.
  // Access to storage->finalizer protected by GIL
  torch::PyObjectFinalizer* finalizer = new torch::PyObjectFinalizer(ref.get());
  std::swap(storage->finalizer, finalizer->next_);
  storage->finalizer.reset(finalizer);

  return ref.release();
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(newWithWeakPtr)(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPObjectPtr ref(PyObject_GetAttrString(arg, "cdata"));
  if (!ref) {
    return NULL;
  } else if (ref.get() == Py_None) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(THPUtils_checkLong(ref.get()),
      "_new_with_weak_ptr(): arg.cdata must be an 'int'");
  THStorage *weak_storage = (THStorage*)PyLong_AsVoidPtr(ref.get());
  if (auto* storage = THStorage_weakLock(weak_storage)) {
    return THPStorage_(New)(storage);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(freeWeakRef)(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  if (arg == Py_None) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(THPUtils_checkLong(arg),
      "_free_weak_ref(): arg must be an 'int'");
  THStorage *weak_storage = (THStorage*)PyLong_AsVoidPtr(arg);
  THStorage_weakFree(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(sharedFd)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THMapAllocatorContext *ctx = NULL;
#ifndef THC_GENERIC_FILE
  THWStorage *storage = self->cdata;
  if (storage->allocatorVoidPtr == &THMapAllocator) {
    ctx = (THMapAllocatorContext*)storage->allocatorContext;
  }
#endif

  THPUtils_assert(ctx, "couldn't retrieve a shared file descriptor");
  return PyLong_FromLong(THMapAllocatorContext_fd(ctx));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newView)(THPStorage *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_Size(args) != 2 || !THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))
      || ! THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "_new_view", 1, "(int offset, int size)");
    return NULL;
  }
  int64_t offset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
  int64_t size = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  return THPStorage_(newTHView)(self->cdata, offset, size);
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(isShared)(THPStorage *self)
{
#ifdef THC_GENERIC_FILE
  Py_RETURN_TRUE;
#else
  void *allocator = self->cdata->allocatorVoidPtr;
  if (allocator == &THMapAllocator ||
      allocator == &THManagedSharedAllocator ||
      // This is a bit racy...
      self->cdata->weakcount.load() > 1) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
#endif
}

static PyMethodDef THPStorage_(sharingMethods)[] = {
  {"_new_with_weak_ptr", (PyCFunction)THPStorage_(newWithWeakPtr), METH_O | METH_CLASS, NULL},
#ifdef THC_GENERIC_FILE
  {"_share_cuda_", (PyCFunction)THPStorage_(shareCuda), METH_NOARGS, NULL},
  {"_new_shared_cuda", (PyCFunction)THPStorage_(newSharedCuda), METH_VARARGS | METH_STATIC, NULL},
#else
  {"_share_fd_", (PyCFunction)THPStorage_(shareFd), METH_NOARGS, NULL},
  {"_new_shared_fd", (PyCFunction)THPStorage_(newSharedFd), METH_VARARGS | METH_STATIC, NULL},
  {"_new_using_fd", (PyCFunction)THPStorage_(pyNewFdStorage), METH_VARARGS | METH_STATIC, NULL},
  {"_share_filename_", (PyCFunction)THPStorage_(shareFilename), METH_NOARGS, NULL},
  {"_new_shared_filename", (PyCFunction)THPStorage_(newSharedFilename), METH_VARARGS | METH_STATIC, NULL},
  {"_new_using_filename", (PyCFunction)THPStorage_(pyNewFilenameStorage), METH_VARARGS | METH_STATIC, NULL},
#endif
  {"_weak_ref", (PyCFunction)THPStorage_(weakRef), METH_O, NULL},
  {"_free_weak_ref", (PyCFunction)THPStorage_(freeWeakRef), METH_O | METH_STATIC, NULL},
  {"_new_view", (PyCFunction)THPStorage_(newView), METH_VARARGS, NULL},
  {"_shared_decref", (PyCFunction)THPStorage_(sharedDecref), METH_NOARGS, NULL},
  {"_shared_incref", (PyCFunction)THPStorage_(sharedIncref), METH_NOARGS, NULL},
  {"_get_shared_fd", (PyCFunction)THPStorage_(sharedFd), METH_NOARGS, NULL},
  {"is_shared", (PyCFunction)THPStorage_(isShared), METH_NOARGS, NULL},
  {NULL}
};

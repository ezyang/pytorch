#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include "err.h"
#include "socket.h"
#include "libshm.h"

std::unordered_map<std::string, ClientSocket> managers;
std::string manager_executable_path;

AllocInfo get_alloc_info(THManagedMapAllocator *ctx) {
  AllocInfo info = {0};
  info.pid = getpid();
  info.free = false;
  const char *filename = ctx->filename();
  size_t len = strlen(filename);
  if (len >= sizeof(info.filename)) {
    throw std::runtime_error("THMapAllocatorContext_filename too long");
  }
  memcpy(info.filename, filename, len + 1);
  return info;
}

void start_manager() {
  int pipe_ends[2];
  SYSCHECK(pipe(pipe_ends));

  pid_t pid;
  SYSCHECK(pid = fork());
  if (!pid) {
    close(pipe_ends[0]);
    dup2(pipe_ends[1], 1); // Replace stdout
    close(pipe_ends[1]);
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);
    exit(1);
  }
  SYSCHECK(close(pipe_ends[1]));

  ssize_t bytes_read;
  char buffer[1000];
  std::string handle;
  for (;;) {
    SYSCHECK(bytes_read = read(pipe_ends[0], buffer, sizeof(buffer)));
    handle.append(buffer, bytes_read);
    if (bytes_read == 0 || handle[handle.length() - 1] == '\n') {
      break;
    }
  }
  SYSCHECK(close(pipe_ends[0]));
  if (handle.length() == 0) {
    std::string msg("error executing torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\"";
    throw std::runtime_error(msg);
  }

  handle.pop_back(); // remove \n
  if (handle == "ERROR")
    throw std::exception();

  ClientSocket manager {handle};
  managers.emplace(std::move(handle), std::move(manager));
}

ClientSocket& get_manager_socket(const std::string& manager_handle) {
  auto it = managers.find(manager_handle);
  if (it == managers.end()) {
    auto socket = ClientSocket(manager_handle);
    auto result = managers.emplace(manager_handle, std::move(socket));
    return result.first->second;
  } else {
    return it->second;
  }
}

void libshm_init(const char *manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);
}

void THManagedMapAllocator::initializeManager() {
  // TODO: unlock GIL when contacting the manager
  try {
    ClientSocket *socket;
    if (!manager_handle_.empty()) {
      socket = &get_manager_socket(manager_handle_);
    } else {
      if (managers.size() == 0) {
        start_manager();
      }
      const auto &manager = managers.begin();
      manager_handle_ = manager->first;
      socket = &manager->second;
    }
    AllocInfo info = get_alloc_info(this);
    socket->register_allocation(info);
  } catch(std::exception &e) {
    THError(e.what());
  }
}

THManagedMapAllocator::THManagedMapAllocator(const char *manager_handle, const char *filename, int flags, ptrdiff_t size)
  : THRefcountedMapAllocator(filename, flags, size)
  // TODO: Perhaps it should be an at::optional<std::string>?
  , manager_handle_(manager_handle ? manager_handle : "") {

    initializeManager();
}

THManagedMapAllocator::THManagedMapAllocator(WithFd, const char *manager_handle, const char *filename, int fd, int flags, ptrdiff_t size)
  : THRefcountedMapAllocator(WITH_FD, filename, fd, flags, size)
  , manager_handle_(manager_handle ? manager_handle : "") {

    initializeManager();
}

THManagedMapAllocator::~THManagedMapAllocator() {
  AllocInfo info = get_alloc_info(this);
  info.free = true;
  ClientSocket &socket = get_manager_socket(manager_handle_);
  // TODO: I think its OK to register deallocation before we actually
  // deallocate; I hope so!
  socket.register_deallocation(info);
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::SupervisedPtr THManagedMapAllocator::makeSupervisedPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size) {
  auto* supervisor = new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {supervisor->data(), {supervisor, &deleteTHManagedMapAllocator}};
}

at::SupervisedPtr THManagedMapAllocator::makeSupervisedPtr(WithFd, const char* manager_handle, const char* filename, int fd, int flags, ptrdiff_t size) {
  auto* supervisor = new THManagedMapAllocator(WITH_FD, manager_handle, filename, fd, flags, size);
  return {supervisor->data(), {supervisor, &deleteTHManagedMapAllocator}};
}

THManagedMapAllocator* THManagedMapAllocator::fromSupervisedPtr(const at::SupervisedPtr& sptr) {
  if (sptr.supervisor_.get_deleter() != &deleteTHManagedMapAllocator) return nullptr;
  return static_cast<THManagedMapAllocator*>(sptr.supervisor_.get());
}

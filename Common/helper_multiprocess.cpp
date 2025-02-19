/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "helper_multiprocess.h"
#include <cstdlib>
#include <string>

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;
  info->shmHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                      PAGE_READWRITE, 0, (DWORD)sz, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
#endif
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;

  info->shmHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
#endif
}

void sharedMemoryClose(sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  if (info->addr) {
    UnmapViewOfFile(info->addr);
  }
  if (info->shmHandle) {
    CloseHandle(info->shmHandle);
  }
#else
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
#endif
}

int spawnProcess(Process *process, const char *app, char *const *args) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  STARTUPINFO si = {0};
  BOOL status;
  size_t arglen = 0;
  size_t argIdx = 0;
  std::string arg_string;
  memset(process, 0, sizeof(*process));

  while (*args) {
    arg_string.append(*args).append(1, ' ');
    args++;
  }

  status = CreateProcess(app, LPSTR(arg_string.c_str()), NULL, NULL, FALSE, 0,
                         NULL, NULL, &si, process);

  return status ? 0 : GetLastError();
#else
  *process = fork();
  if (*process == 0) {
    if (0 > execvp(app, args)) {
      return errno;
    }
  } else if (*process < 0) {
    return errno;
  }
  return 0;
#endif
}

int waitProcess(Process *process) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  DWORD exitCode;
  WaitForSingleObject(process->hProcess, INFINITE);
  GetExitCodeProcess(process->hProcess, &exitCode);
  CloseHandle(process->hProcess);
  CloseHandle(process->hThread);
  return (int)exitCode;
#else
  int status = 0;
  do {
    if (0 > waitpid(*process, &status, 0)) {
      return errno;
    }
  } while (!WIFEXITED(status));
  return WEXITSTATUS(status);
#endif
}

#if defined(__linux__) || defined(__QNX__)
int ipcCreateSocket(ipcHandle *&handle, const char *name,
                    const std::vector<Process> &processes) {
  int server_fd;
  struct sockaddr_un servaddr;

  handle = new ipcHandle;
  memset(handle, 0, sizeof(*handle));
  handle->socket = -1;
  handle->socketName = NULL;

  // Creating socket file descriptor
  if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
    perror("IPC failure: Socket creation failed");
    return -1;
  }

  unlink(name);
  bzero(&servaddr, sizeof(servaddr));
  servaddr.sun_family = AF_UNIX;

  size_t len = strlen(name);
  if (len > (sizeof(servaddr.sun_path) - 1)) {
    perror("IPC failure: Cannot bind provided name to socket. Name too large");
    return -1;
  }

  strncpy(servaddr.sun_path, name, len);

  if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
    perror("IPC failure: Binding socket failed");
    return -1;
  }

  handle->socketName = new char[strlen(name) + 1];
  strcpy(handle->socketName, name);
  handle->socket = server_fd;
  return 0;
}

int ipcOpenSocket(ipcHandle *&handle) {
  int sock = 0;
  struct sockaddr_un cliaddr;

  handle = new ipcHandle;
  memset(handle, 0, sizeof(*handle));

  if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    perror("IPC failure:Socket creation error");
    return -1;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;
  char temp[10];

  // Create unique name for the socket.
  sprintf(temp, "%u", getpid());

  strcpy(cliaddr.sun_path, temp);
  if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    perror("IPC failure: Binding socket failed");
    return -1;
  }

  handle->socket = sock;
  handle->socketName = new char[strlen(temp) + 1];
  strcpy(handle->socketName, temp);

  return 0;
}

int ipcCloseSocket(ipcHandle *handle) {
  if (!handle) {
    return -1;
  }

  if (handle->socketName) {
    unlink(handle->socketName);
    delete[] handle->socketName;
  }
  close(handle->socket);
  delete handle;
  return 0;
}

int ipcRecvShareableHandle(ipcHandle *handle, ShareableHandle *shHandle) {
  struct msghdr msg = {0};
  struct iovec iov[1];
  struct cmsghdr cm;

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    // This will not work on QNX as QNX CMSG_SPACE calls __cmsg_alignbytes
    // And __cmsg_alignbytes is a runtime function instead of compile-time macros
    // char control[CMSG_SPACE(sizeof(int))]
    char* control;
  } control_un;

  size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
  control_un.control = (char*) malloc(sizeof_control);
  struct cmsghdr *cmptr;
  ssize_t n;
  int receivedfd;
  char dummy_buffer[1];
  ssize_t sendResult;
  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof_control;

  iov[0].iov_base = (void *)dummy_buffer;
  iov[0].iov_len = sizeof(dummy_buffer);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
    perror("IPC failure: Receiving data over socket failed");
    free(control_un.control);
    return -1;
  }

  if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
      (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
      free(control_un.control);
      return -1;
    }

    memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
    *(int *)shHandle = receivedfd;
  } else {
    free(control_un.control);
    return -1;
  }

  free(control_un.control);
  return 0;
}

int ipcRecvDataFromClient(ipcHandle *serverHandle, void *data, size_t size) {
  ssize_t readResult;
  struct sockaddr_un cliaddr;
  socklen_t len = sizeof(cliaddr);

  readResult = recvfrom(serverHandle->socket, data, size, 0,
                        (struct sockaddr *)&cliaddr, &len);
  if (readResult == -1) {
    perror("IPC failure: Receiving data over socket failed");
    return -1;
  }
  return 0;
}

int ipcSendDataToServer(ipcHandle *handle, const char *serverName,
                        const void *data, size_t size) {
  ssize_t sendResult;
  struct sockaddr_un serveraddr;

  bzero(&serveraddr, sizeof(serveraddr));
  serveraddr.sun_family = AF_UNIX;
  strncpy(serveraddr.sun_path, serverName, sizeof(serveraddr.sun_path) - 1);

  sendResult = sendto(handle->socket, data, size, 0,
                      (struct sockaddr *)&serveraddr, sizeof(serveraddr));
  if (sendResult <= 0) {
    perror("IPC failure: Sending data over socket failed");
  }

  return 0;
}

int ipcSendShareableHandle(ipcHandle *handle,
                           const std::vector<ShareableHandle> &shareableHandles,
                           Process process, int data) {
  struct msghdr msg;
  struct iovec iov[1];

  union {
    struct cmsghdr cm;
    char* control;
  } control_un;

  size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
  control_un.control = (char*) malloc(sizeof_control);

  struct cmsghdr *cmptr;
  ssize_t readResult;
  struct sockaddr_un cliaddr;
  socklen_t len = sizeof(cliaddr);

  // Construct client address to send this SHareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;
  char temp[10];
  sprintf(temp, "%u", process);
  strcpy(cliaddr.sun_path, temp);
  len = sizeof(cliaddr);

  // Send corresponding shareable handle to the client
  int sendfd = (int)shareableHandles[data];

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof_control;

  cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;

  memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  iov[0].iov_base = (void *)"";
  iov[0].iov_len = 1;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
  if (sendResult <= 0) {
    perror("IPC failure: Sending data over socket failed");
    free(control_un.control);
    return -1;
  }

  free(control_un.control);
  return 0;
}

int ipcSendShareableHandles(
    ipcHandle *handle, const std::vector<ShareableHandle> &shareableHandles,
    const std::vector<Process> &processes) {
  // Send all shareable handles to every single process.
  for (int i = 0; i < shareableHandles.size(); i++) {
    for (int j = 0; j < processes.size(); j++) {
      checkIpcErrors(
          ipcSendShareableHandle(handle, shareableHandles, processes[j], i));
    }
  }
  return 0;
}

int ipcRecvShareableHandles(ipcHandle *handle,
                            std::vector<ShareableHandle> &shareableHandles) {
  for (int i = 0; i < shareableHandles.size(); i++) {
    checkIpcErrors(ipcRecvShareableHandle(handle, &shareableHandles[i]));
  }
  return 0;
}

int ipcCloseShareableHandle(ShareableHandle shHandle) {
  return close(shHandle);
}

#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Generic name to build individual Mailslot names by appending process ids.
LPTSTR SlotName = (LPTSTR)TEXT("\\\\.\\mailslot\\sample_mailslot_");

int ipcCreateSocket(ipcHandle *&handle, const char *name,
                    const std::vector<Process> &processes) {
  handle = new ipcHandle;
  handle->hMailslot.resize(processes.size());

  // Open Mailslots of all clients and store respective handles.
  for (int i = 0; i < handle->hMailslot.size(); ++i) {
    std::basic_string<TCHAR> childSlotName(SlotName);
    char tempBuf[20];
    _itoa_s(processes[i].dwProcessId, tempBuf, 10);
    childSlotName += TEXT(tempBuf);

    HANDLE hFile =
        CreateFile(TEXT(childSlotName.c_str()), GENERIC_WRITE, FILE_SHARE_READ,
                   (LPSECURITY_ATTRIBUTES)NULL, OPEN_EXISTING,
                   FILE_ATTRIBUTE_NORMAL, (HANDLE)NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      printf("IPC failure: Opening Mailslot by CreateFile failed with %d\n",
             GetLastError());
      return -1;
    }
    handle->hMailslot[i] = hFile;
  }
  return 0;
}

int ipcOpenSocket(ipcHandle *&handle) {
  handle = new ipcHandle;
  HANDLE hSlot;

  std::basic_string<TCHAR> clientSlotName(SlotName);
  char tempBuf[20];
  _itoa_s(GetCurrentProcessId(), tempBuf, 10);
  clientSlotName += TEXT(tempBuf);

  hSlot = CreateMailslot((LPSTR)clientSlotName.c_str(), 0,
                         MAILSLOT_WAIT_FOREVER, (LPSECURITY_ATTRIBUTES)NULL);
  if (hSlot == INVALID_HANDLE_VALUE) {
    printf("IPC failure: CreateMailslot failed for client with %d\n",
           GetLastError());
    return -1;
  }

  handle->hMailslot.push_back(hSlot);
  return 0;
}

int ipcSendData(HANDLE mailslot, const void *data, size_t sz) {
  BOOL result;
  DWORD cbWritten;

  result = WriteFile(mailslot, data, (DWORD)sz, &cbWritten, (LPOVERLAPPED)NULL);
  if (!result) {
    printf("IPC failure: WriteFile failed with %d.\n", GetLastError());
    return -1;
  }
  return 0;
}

int ipcRecvData(ipcHandle *handle, void *data, size_t sz) {
  DWORD cbRead = 0;

  if (!ReadFile(handle->hMailslot[0], data, (DWORD)sz, &cbRead, NULL)) {
    printf("IPC failure: ReadFile failed with %d.\n", GetLastError());
    return -1;
  }

  if (sz != (size_t)cbRead) {
    printf(
        "IPC failure: ReadFile didn't receive the expected number of bytes\n");
    return -1;
  }

  return 0;
}

int ipcSendShareableHandles(
    ipcHandle *handle, const std::vector<ShareableHandle> &shareableHandles,
    const std::vector<Process> &processes) {
  // Send all shareable handles to every single process.
  for (int i = 0; i < processes.size(); i++) {
    HANDLE hProcess =
        OpenProcess(PROCESS_DUP_HANDLE, FALSE, processes[i].dwProcessId);
    if (hProcess == INVALID_HANDLE_VALUE) {
      printf("IPC failure: OpenProcess failed (%d)\n", GetLastError());
      return -1;
    }

    for (int j = 0; j < shareableHandles.size(); j++) {
      HANDLE hDup = INVALID_HANDLE_VALUE;
      // Duplicate the handle into the target process's space
      if (!DuplicateHandle(GetCurrentProcess(), shareableHandles[j], hProcess,
                           &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        printf("IPC failure: DuplicateHandle failed (%d)\n", GetLastError());
        return -1;
      }
      checkIpcErrors(ipcSendData(handle->hMailslot[i], &hDup, sizeof(hDup)));
    }
    CloseHandle(hProcess);
  }
  return 0;
}

int ipcRecvShareableHandles(ipcHandle *handle,
                            std::vector<ShareableHandle> &shareableHandles) {
  for (int i = 0; i < shareableHandles.size(); i++) {
    checkIpcErrors(
        ipcRecvData(handle, &shareableHandles[i], sizeof(shareableHandles[i])));
  }
  return 0;
}

int ipcCloseSocket(ipcHandle *handle) {
  for (int i = 0; i < handle->hMailslot.size(); i++) {
    CloseHandle(handle->hMailslot[i]);
  }
  delete handle;
  return 0;
}

int ipcCloseShareableHandle(ShareableHandle shHandle) {
  CloseHandle(shHandle);
  return 0;
}

#endif

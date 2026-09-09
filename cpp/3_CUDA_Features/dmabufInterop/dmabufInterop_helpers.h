/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * POSIX helpers used by the cross-process demos in dmabufInterop.cu:
 *   - sendFd / recvFd: transport a file descriptor between two processes
 *     over a UNIX domain socket via the SCM_RIGHTS control message.
 *   - reapCrossProcessChild: waitpid() wrapper that returns 0 iff the child
 *     exited cleanly with status 0.
 *
 * These are POSIX-only (no CUDA code) and are separated out so that
 * dmabufInterop.cu stays focused on the CUDA dma-buf export/import flow.
 */

#ifndef DMABUF_INTEROP_HELPERS_H
#define DMABUF_INTEROP_HELPERS_H

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

/* Send a file descriptor over a UNIX domain socket using SCM_RIGHTS. */
static inline void sendFd(int sock, int fd)
{
    struct msghdr msg = {};
    char dummy = 'F';
    struct iovec iov = { &dummy, 1 };
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    char cbuf[CMSG_SPACE(sizeof(int))] = {};
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

    (void)sendmsg(sock, &msg, 0);
}

/* Receive a file descriptor from a UNIX domain socket. Returns -1 if no
 * valid single-fd SCM_RIGHTS control message was received. */
static inline int recvFd(int sock)
{
    struct msghdr msg = {};
    char dummy = 0;
    struct iovec iov = { &dummy, 1 };
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    char cbuf[CMSG_SPACE(sizeof(int))] = {};
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);

    (void)recvmsg(sock, &msg, 0);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    /* Validate that recvmsg actually delivered a single-fd SCM_RIGHTS control
     * message before dereferencing -- a truncated or absent ancillary buffer
     * would otherwise crash on the memcpy below. */
    if (cmsg == NULL || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) return -1;
    int fd = -1;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

/* Wait for a child process to exit and return 0 iff it exited cleanly with
 * status 0. */
static inline int reapCrossProcessChild(pid_t pid)
{
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }
    return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : -1;
}

#endif /* DMABUF_INTEROP_HELPERS_H */

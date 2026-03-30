#ifndef CUDA_SAMPLES_PTHREAD_CLOCK_COMPAT_H_
#define CUDA_SAMPLES_PTHREAD_CLOCK_COMPAT_H_

#include <pthread.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int pthread_cond_clockwait(pthread_cond_t *__restrict cond,
                           pthread_mutex_t *__restrict mutex,
                           clockid_t clock_id,
                           const struct timespec *__restrict abstime);

int pthread_mutex_clocklock(pthread_mutex_t *__restrict mutex,
                            clockid_t clock_id,
                            const struct timespec *__restrict abstime);

int pthread_rwlock_clockrdlock(pthread_rwlock_t *__restrict rwlock,
                               clockid_t clock_id,
                               const struct timespec *__restrict abstime);

int pthread_rwlock_clockwrlock(pthread_rwlock_t *__restrict rwlock,
                               clockid_t clock_id,
                               const struct timespec *__restrict abstime);

#ifdef __cplusplus
}
#endif

#endif

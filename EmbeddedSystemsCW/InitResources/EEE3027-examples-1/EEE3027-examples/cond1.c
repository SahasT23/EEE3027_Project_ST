#include <stdio.h>
#include <pthread.h>

pthread_mutex_t condition_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condition_cond = PTHREAD_COND_INITIALIZER;

void *t_func1 ()
{
  system ("sleep 1");
  pthread_mutex_lock (&condition_mutex);
  printf ("thread 1: waiting started\n");
  pthread_cond_wait (&condition_cond, &condition_mutex);
  printf ("thread 1: waiting finished\n");
  pthread_mutex_unlock (&condition_mutex);
}

void *t_func2 ()
{
  system ("sleep 2"); // 0 - deadlock, 2 - no deadlock
  printf ("thread 2: locking mutex\n");
  pthread_mutex_lock (&condition_mutex);
  printf ("thread 2: signal\n");
  pthread_cond_signal (&condition_cond);
  pthread_mutex_unlock (&condition_mutex);
//  system ("sleep 1");
  printf ("thread 2: mutex unlocked\n");
}

main ()
{
  pthread_t thread1, thread2;

  pthread_create (&thread1, NULL, &t_func1, NULL);
  pthread_create (&thread2, NULL, &t_func2, NULL);
  pthread_join (thread1, NULL);
  pthread_join (thread2, NULL);
}



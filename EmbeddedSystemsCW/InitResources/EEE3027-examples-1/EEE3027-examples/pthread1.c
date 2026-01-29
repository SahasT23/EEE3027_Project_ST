#include <stdio.h>
#include <pthread.h>

void
thread_func (void *ptr)
{
  printf ("%s \n", (char *) ptr);
}

main ()
{
  pthread_t t1, t2;
  char *msg1 = "Thread 1";
  char *msg2 = "Thread 2";
  int t1_ret, t2_ret;

  t1_ret = pthread_create (&t1, NULL, (void *) &thread_func, (void *) msg1);
  t2_ret = pthread_create (&t2, NULL, (void *) &thread_func, (void *) msg2);

  pthread_join (t1, NULL);
  pthread_join (t2, NULL);

  printf ("Threads finished with %d/%d codes\n", t1_ret, t2_ret);
  
  exit (0);
}

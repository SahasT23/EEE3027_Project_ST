#include <stdio.h>
#include <pthread.h>

int counter = 0;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

void
thread_func (void *ptr)
{
  int i;
  for (i=0;i<100000;i++)
    {
      //pthread_mutex_lock (&mutex1);
      counter++;
      //pthread_mutex_unlock (&mutex1);
    }
  printf ("Thread %s, counter = %d \n", ptr, counter);
}

main ()
{
  pthread_t t1, t2;
  int t_id1, t_id2;
  char *msg1 = "Thread 1";
  char *msg2 = "Thread 2";


  t_id1 = pthread_create (&t1, NULL, (void *) &thread_func,  (void *) msg1);
  t_id2 = pthread_create (&t2, NULL, (void *) &thread_func,  (void *) msg2);

  pthread_join (t1, NULL);
  pthread_join (t2, NULL);
}



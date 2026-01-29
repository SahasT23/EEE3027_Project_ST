#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

void
thread_func (void *ptr)
{
  int i;
  for (i = 0; i < 10; i++)
    {
      pthread_mutex_lock (&mutex1);
      printf ("%d", *(int*)ptr);
      pthread_mutex_unlock (&mutex1);
      system("sleep 0.1"); 
    }
}

main ()
{
  pthread_t t1, t2;
  int t_id1, t_id2, a=1, b=2;

  t_id1 = pthread_create (&t1, NULL, (void *) &thread_func, &a);
  t_id2 = pthread_create (&t2, NULL, (void *) &thread_func, &b);

  pthread_join (t1, NULL);
  pthread_join (t2, NULL);

  exit (0);
}

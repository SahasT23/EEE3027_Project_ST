#include <stdio.h>
#include <pthread.h>
void t_function1 ()
{
  usleep (1000);
  printf ("Hello\n");
  fflush(stdout);
  pthread_exit (0);
}
void t_function2 ()
{
  usleep (1000);
  printf ("World\n");
  fflush(stdout);
  pthread_exit (0);
}
int main ()
{
  pthread_t th1, th2;
  pthread_create (&th1, NULL, (void *) &t_function1, NULL);
  pthread_create (&th1, NULL, (void *) &t_function2, NULL);
  usleep (1000);
  exit (0);
}

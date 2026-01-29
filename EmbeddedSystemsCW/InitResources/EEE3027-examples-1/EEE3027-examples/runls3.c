#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main ()
{
  pid_t pid;

  switch (pid = fork ())
    {
    case -1:
      perror ("fork error");
      exit (1);
    case 0:
      execl ("/bin/ls", "ls", "-l", (char *) 0);
      perror ("exec error");
      exit (1);
    default:
      wait ((int *) 0);
      printf ("program ls finished\n");
      exit (0);
    }
}




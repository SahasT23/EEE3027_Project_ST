# Define functions
process_model()
 {
  xterm -e "echo $1 ; echo press Ctl-c to finish ; sleep 1h"
  return 0
 }

# Start process_1 first
process_model process_1

# Once it has finished choose process_2 or process_3
PS1="[type 'exit 0' or 'exit 1' followed by enter] " bash

if [ "$?" == 0 ] ;
 then process_model process_2 ;
 else process_model process_3 ;
fi

# After the chosen process has finished start process_4
process_model process_4


